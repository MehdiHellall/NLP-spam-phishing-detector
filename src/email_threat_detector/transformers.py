"""Optional Hugging Face transformer training workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from email_threat_detector.constants import DEFAULT_RANDOM_SEED, LABEL_NAMES, LABEL_TO_ID
from email_threat_detector.metrics import compute_classification_metrics


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configuration for a small local transformer run."""

    split_dir: Path
    output_dir: Path = Path("artifacts/transformers/bert-tiny")
    metrics_path: Path | None = None
    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    max_length: int = 128
    epochs: float = 1.0
    batch_size: int = 8
    learning_rate: float = 5e-5
    random_state: int = DEFAULT_RANDOM_SEED


def run_transformer_training(config: TransformerTrainingConfig) -> dict[str, Any]:
    """Fine-tune a small transformer using prepared CSV splits.

    This function imports Hugging Face dependencies lazily so the package and
    tests remain lightweight when transformer extras are not installed.
    """
    data_files = {
        "train": config.split_dir / "train.csv",
        "validation": config.split_dir / "validation.csv",
        "test": config.split_dir / "test.csv",
    }
    missing_files = [str(path) for path in data_files.values() if not path.is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing prepared transformer split file(s): {missing}")

    try:
        import numpy as np
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer training requires optional dependencies. "
            'Install them with: pip install -e ".[transformers]"'
        ) from exc

    set_seed(config.random_state)

    dataset = load_dataset("csv", data_files={name: str(path) for name, path in data_files.items()})
    dataset = dataset.map(lambda row: {"labels": LABEL_TO_ID[row["label"]]})

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    tokenized = dataset.map(tokenize, batched=True)
    removable_columns = [
        column for column in ["text", "label"] if column in tokenized["train"].column_names
    ]
    tokenized = tokenized.remove_columns(removable_columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABEL_NAMES),
        id2label={index: label for index, label in enumerate(LABEL_NAMES)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions = np.argmax(eval_prediction.predictions, axis=1)
        y_true = [LABEL_NAMES[index] for index in eval_prediction.label_ids]
        y_pred = [LABEL_NAMES[index] for index in predictions]
        metrics = compute_classification_metrics(y_true, y_pred)
        return {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=config.random_state,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    metrics_path = config.metrics_path or config.output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = config.split_dir / "manifest.json"
    split_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    payload = {
        "model_name": config.model_name,
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length,
            "random_state": config.random_state,
        },
        "split_manifest": split_manifest,
        "metrics": evaluation,
        "notes": [
            "Transformer model was fine-tuned on prepared balanced real-data split CSVs.",
            "Classifier head is initialized for the three project labels.",
        ],
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model_name": config.model_name,
        "output_dir": str(config.output_dir),
        "metrics_path": str(metrics_path),
        "metrics": payload["metrics"],
    }
