"""Baseline training and evaluation workflow."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from email_threat_detector.cleaning import clean_messages
from email_threat_detector.constants import (
    DEFAULT_DATASET_SLUG,
    DEFAULT_RANDOM_SEED,
    LABEL_COLUMN,
    TEXT_COLUMN,
)
from email_threat_detector.data import load_messages_csv
from email_threat_detector.duplicates import audit_duplicate_texts, prepare_modeling_data
from email_threat_detector.inference import save_model
from email_threat_detector.metrics import compute_classification_metrics
from email_threat_detector.models import build_baseline_model
from email_threat_detector.sampling import balanced_sample
from email_threat_detector.splits import (
    assert_no_text_overlap,
    stratified_train_validation_test_split,
)


@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration for deterministic modeling data preparation."""

    validation_size: float = 0.1
    test_size: float = 0.1
    balance_classes: bool = True
    samples_per_class: int | None = None
    random_state: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Configuration for a leakage-safe baseline training run."""

    data_path: Path
    model_name: str = "tfidf_logreg"
    reports_dir: Path = Path("reports")
    model_path: Path | None = None
    data: DataPreparationConfig = field(default_factory=DataPreparationConfig)


def prepare_splits(
    frame: pd.DataFrame,
    *,
    config: DataPreparationConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Clean, deconflict, optionally balance, and split a message dataframe."""
    if config is None:
        config = DataPreparationConfig()

    cleaned = clean_messages(frame)
    duplicate_audit = audit_duplicate_texts(cleaned)
    conflict_count = int(duplicate_audit["is_conflict"].sum()) if not duplicate_audit.empty else 0
    duplicate_text_count = (
        int((duplicate_audit["row_count"] > 1).sum()) if not duplicate_audit.empty else 0
    )

    modeling = prepare_modeling_data(cleaned)
    if config.balance_classes:
        modeling = balanced_sample(
            modeling,
            samples_per_class=config.samples_per_class,
            random_state=config.random_state,
        )

    splits = stratified_train_validation_test_split(
        modeling,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    assert_no_text_overlap(splits)

    summary = {
        "input_rows": int(len(frame)),
        "clean_rows": int(len(cleaned)),
        "modeling_rows": int(len(modeling)),
        "duplicate_text_groups": duplicate_text_count,
        "conflicting_text_groups_removed": conflict_count,
        "class_counts": {
            split_name: split[LABEL_COLUMN].value_counts().sort_index().astype(int).to_dict()
            for split_name, split in splits.items()
        },
        "data_config": asdict(config),
    }
    return splits, summary


def train_baseline_model(
    splits: dict[str, pd.DataFrame],
    *,
    model_name: str,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Any:
    """Fit a named sklearn baseline on the training split only."""
    model = build_baseline_model(model_name, random_state=random_state)
    train = splits["train"]
    model.fit(train[TEXT_COLUMN], train[LABEL_COLUMN])
    return model


def evaluate_model(model: Any, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Evaluate a fitted model on every non-empty non-training split."""
    evaluations: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        split = splits.get(split_name)
        if split is None or split.empty:
            continue
        predictions = model.predict(split[TEXT_COLUMN])
        evaluations[split_name] = compute_classification_metrics(split[LABEL_COLUMN], predictions)
    return evaluations


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write indented JSON to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dataset_reference(path: str | Path) -> dict[str, str]:
    """Return a publishable dataset reference without local absolute paths."""
    return {
        "filename": Path(path).name,
        "source": DEFAULT_DATASET_SLUG,
    }


def publishable_path(path: str | Path) -> str:
    """Return a path string that avoids leaking absolute local directories."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.name
    return str(candidate)


def maybe_write_confusion_matrix_plot(
    metrics: dict[str, Any],
    *,
    labels: list[str],
    title: str,
    path: Path,
) -> bool:
    """Write a confusion-matrix figure when matplotlib is installed."""
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "email-threat-detector-matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, str(value), ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def run_baseline_training(config: BaselineTrainingConfig) -> dict[str, Any]:
    """Run the full leakage-safe baseline workflow and persist outputs."""
    frame = load_messages_csv(config.data_path)
    splits, data_summary = prepare_splits(frame, config=config.data)
    model = train_baseline_model(
        splits,
        model_name=config.model_name,
        random_state=config.data.random_state,
    )
    evaluations = evaluate_model(model, splits)

    reports_dir = config.reports_dir
    metrics_path = reports_dir / "metrics" / f"{config.model_name}_metrics.json"
    payload = {
        "model_name": config.model_name,
        "dataset": dataset_reference(config.data_path),
        "data_summary": data_summary,
        "metrics": evaluations,
        "notes": [
            "Vectorizers are fit inside sklearn Pipelines on the training split only.",
            (
                "Rows with duplicate normalized text and conflicting labels are removed "
                "before splitting."
            ),
            (
                "Metrics are reproducible only for the exact data file and configuration "
                "recorded here."
            ),
        ],
    }
    write_json(payload, metrics_path)

    figure_written = False
    if "test" in evaluations:
        figure_written = maybe_write_confusion_matrix_plot(
            evaluations["test"],
            labels=sorted(splits["train"][LABEL_COLUMN].unique().tolist()),
            title=f"{config.model_name} test confusion matrix",
            path=reports_dir / "figures" / f"{config.model_name}_test_confusion_matrix.png",
        )

    model_path = config.model_path
    if model_path is not None:
        save_model(
            model,
            model_path,
            metadata={
                "model_name": config.model_name,
                "data_summary": data_summary,
                "metrics_file": metrics_path.name,
            },
        )

    return {
        "metrics_path": str(metrics_path),
        "model_path": str(model_path) if model_path is not None else None,
        "figure_written": figure_written,
        **payload,
    }


def write_transformer_split_files(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    config: DataPreparationConfig | None = None,
) -> dict[str, Any]:
    """Create deterministic CSV split files for transformer experiments."""
    if config is None:
        config = DataPreparationConfig()

    frame = load_messages_csv(data_path)
    splits, data_summary = prepare_splits(frame, config=config)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, split in splits.items():
        split.to_csv(destination / f"{split_name}.csv", index=False)

    manifest = {
        "dataset": dataset_reference(data_path),
        "output_dir": publishable_path(destination),
        "data_summary": data_summary,
        "notes": [
            (
                "These split files use the same cleanup, conflict removal, balancing, "
                "and split logic as the baseline pipeline."
            ),
            "Use these files as the input contract for transformer fine-tuning.",
        ],
    }
    write_json(manifest, destination / "manifest.json")
    return manifest
