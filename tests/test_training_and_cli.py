import json

import pandas as pd
import pytest

from email_threat_detector.cli import main
from email_threat_detector.training import (
    BaselineTrainingConfig,
    DataPreparationConfig,
    dataset_reference,
    evaluate_model,
    maybe_write_confusion_matrix_plot,
    prepare_splits,
    publishable_path,
    run_baseline_training,
    write_transformer_split_files,
)
from email_threat_detector.transformers import TransformerTrainingConfig, run_transformer_training


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text": "team project agenda", "label": "ham"},
            {"text": "family dinner update", "label": "ham"},
            {"text": "office meeting notes", "label": "ham"},
            {"text": "verify account password", "label": "phish"},
            {"text": "bank login suspended", "label": "phish"},
            {"text": "confirm wallet credentials", "label": "phish"},
            {"text": "free prize coupon", "label": "spam"},
            {"text": "promo discount offer", "label": "spam"},
            {"text": "winner cash deal", "label": "spam"},
            {"text": "duplicate conflict", "label": "ham"},
            {"text": " duplicate   conflict ", "label": "spam"},
        ]
    )


def _write_csv(path, frame: pd.DataFrame | None = None) -> None:
    (frame if frame is not None else _training_frame()).to_csv(path, index=False)


def test_prepare_splits_removes_conflicts_balances_and_tracks_counts() -> None:
    splits, summary = prepare_splits(
        _training_frame(),
        config=DataPreparationConfig(validation_size=1 / 3, test_size=1 / 3, random_state=1),
    )

    assert summary["input_rows"] == 11
    assert summary["conflicting_text_groups_removed"] == 1
    assert summary["modeling_rows"] == 9
    assert {name: len(split) for name, split in splits.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }


def test_prepare_splits_uses_default_config_when_not_supplied() -> None:
    frame = pd.DataFrame(
        [
            {"text": f"{label} default message {index}", "label": label}
            for label in ("ham", "phish", "spam")
            for index in range(12)
        ]
    )

    splits, summary = prepare_splits(frame)

    assert summary["input_rows"] == 36
    assert set(splits) == {"train", "validation", "test"}


def test_run_baseline_training_writes_publishable_metrics_and_model(tmp_path) -> None:
    data_path = tmp_path / "messages.csv"
    reports_dir = tmp_path / "reports"
    model_path = tmp_path / "models" / "baseline.joblib"
    _write_csv(data_path)

    result = run_baseline_training(
        BaselineTrainingConfig(
            data_path=data_path,
            model_name="tfidf_complement_nb",
            reports_dir=reports_dir,
            model_path=model_path,
            data=DataPreparationConfig(validation_size=1 / 3, test_size=1 / 3, random_state=2),
        )
    )

    metrics_path = reports_dir / "metrics" / "tfidf_complement_nb_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert result["metrics_path"] == str(metrics_path)
    assert model_path.is_file()
    assert payload["dataset"] == dataset_reference(data_path)
    assert "test" in payload["metrics"]
    assert str(tmp_path) not in json.dumps(payload)


def test_write_transformer_split_files_writes_manifest_and_csvs(tmp_path) -> None:
    data_path = tmp_path / "messages.csv"
    output_dir = tmp_path / "splits"
    _write_csv(data_path)

    manifest = write_transformer_split_files(
        data_path,
        output_dir,
        config=DataPreparationConfig(validation_size=1 / 3, test_size=1 / 3, random_state=3),
    )

    assert manifest["output_dir"] == "splits"
    assert (output_dir / "manifest.json").is_file()
    assert {path.name for path in output_dir.glob("*.csv")} == {
        "train.csv",
        "validation.csv",
        "test.csv",
    }


def test_evaluate_model_skips_empty_splits_and_plot_is_optional(tmp_path) -> None:
    class ConstantModel:
        def predict(self, texts):
            return ["ham" for _ in texts]

    splits = {
        "validation": pd.DataFrame({"text": [], "label": []}),
        "test": pd.DataFrame({"text": ["hello"], "label": ["ham"]}),
    }

    metrics = evaluate_model(ConstantModel(), splits)
    assert sorted(metrics) == ["test"]
    figure_path = tmp_path / "figure.png"
    figure_written = maybe_write_confusion_matrix_plot(
        metrics["test"],
        labels=["ham", "phish", "spam"],
        title="test",
        path=figure_path,
    )
    assert figure_written == figure_path.exists()


def test_publishable_path_preserves_relative_paths_and_redacts_absolute(tmp_path) -> None:
    assert publishable_path("reports/metrics").replace("\\", "/") == "reports/metrics"
    assert publishable_path(tmp_path / "absolute-output") == "absolute-output"


def test_cli_train_baseline_and_prepare_splits_commands(tmp_path, capsys) -> None:
    data_path = tmp_path / "messages.csv"
    reports_dir = tmp_path / "reports"
    splits_dir = tmp_path / "splits"
    _write_csv(data_path)

    assert (
        main(
            [
                "train-baseline",
                "--data-path",
                str(data_path),
                "--reports-dir",
                str(reports_dir),
                "--validation-size",
                "0.3333333333",
                "--test-size",
                "0.3333333333",
                "--model-name",
                "tfidf_complement_nb",
            ]
        )
        == 0
    )
    assert "metrics:" in capsys.readouterr().out

    assert (
        main(
            [
                "prepare-transformer-splits",
                "--data-path",
                str(data_path),
                "--output-dir",
                str(splits_dir),
                "--validation-size",
                "0.3333333333",
                "--test-size",
                "0.3333333333",
            ]
        )
        == 0
    )
    assert str(splits_dir) in capsys.readouterr().out


def test_cli_train_transformer_reports_missing_split_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing prepared transformer split"):
        main(["train-transformer", "--split-dir", str(tmp_path)])


def test_transformer_training_reports_missing_split_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing prepared transformer split"):
        run_transformer_training(TransformerTrainingConfig(split_dir=tmp_path))
