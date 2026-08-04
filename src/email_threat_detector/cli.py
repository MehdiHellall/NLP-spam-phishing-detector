"""Command line entry points for reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from email_threat_detector.models import BASELINE_MODEL_BUILDERS
from email_threat_detector.training import (
    BaselineTrainingConfig,
    DataPreparationConfig,
    run_baseline_training,
    write_transformer_split_files,
)
from email_threat_detector.transformers import TransformerTrainingConfig, run_transformer_training


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(prog="email-threat-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "train-baseline",
        help="Train a leakage-safe sklearn baseline.",
    )
    baseline.add_argument("--data-path", required=True, type=Path)
    baseline.add_argument(
        "--model-name",
        choices=sorted(BASELINE_MODEL_BUILDERS),
        default="tfidf_logreg",
    )
    baseline.add_argument("--reports-dir", type=Path, default=Path("reports"))
    baseline.add_argument("--model-path", type=Path)
    baseline.add_argument("--validation-size", type=float, default=0.1)
    baseline.add_argument("--test-size", type=float, default=0.1)
    baseline.add_argument("--samples-per-class", type=int)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip deterministic downsampling to the smallest class.",
    )

    splits = subparsers.add_parser(
        "prepare-transformer-splits",
        help="Write deterministic CSV splits for transformer training.",
    )
    splits.add_argument("--data-path", required=True, type=Path)
    splits.add_argument("--output-dir", required=True, type=Path)
    splits.add_argument("--validation-size", type=float, default=0.1)
    splits.add_argument("--test-size", type=float, default=0.1)
    splits.add_argument("--samples-per-class", type=int)
    splits.add_argument("--seed", type=int, default=42)
    splits.add_argument("--no-balance", action="store_true")

    transformer = subparsers.add_parser(
        "train-transformer",
        help="Fine-tune a small Hugging Face model from prepared split CSVs.",
    )
    transformer.add_argument("--split-dir", required=True, type=Path)
    transformer.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/transformers/bert-tiny"),
    )
    transformer.add_argument("--metrics-path", type=Path)
    transformer.add_argument(
        "--model-name",
        default="google/bert_uncased_L-2_H-128_A-2",
    )
    transformer.add_argument("--max-length", type=int, default=128)
    transformer.add_argument("--epochs", type=float, default=1.0)
    transformer.add_argument("--batch-size", type=int, default=8)
    transformer.add_argument("--learning-rate", type=float, default=5e-5)
    transformer.add_argument("--seed", type=int, default=42)
    return parser


def data_config_from_args(args: argparse.Namespace) -> DataPreparationConfig:
    """Translate CLI flags into data-preparation configuration."""
    return DataPreparationConfig(
        validation_size=args.validation_size,
        test_size=args.test_size,
        balance_classes=not args.no_balance,
        samples_per_class=args.samples_per_class,
        random_state=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Run a command and print the primary artifact path."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-baseline":
        result = run_baseline_training(
            BaselineTrainingConfig(
                data_path=args.data_path,
                model_name=args.model_name,
                reports_dir=args.reports_dir,
                model_path=args.model_path,
                data=data_config_from_args(args),
            )
        )
        print(f"metrics: {result['metrics_path']}")
        if result["model_path"]:
            print(f"model: {result['model_path']}")
        return 0

    if args.command == "prepare-transformer-splits":
        result = write_transformer_split_files(
            args.data_path,
            args.output_dir,
            config=data_config_from_args(args),
        )
        print(f"splits: {args.output_dir}")
        return 0

    if args.command == "train-transformer":
        result = run_transformer_training(
            TransformerTrainingConfig(
                split_dir=args.split_dir,
                output_dir=args.output_dir,
                metrics_path=args.metrics_path,
                model_name=args.model_name,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_state=args.seed,
            )
        )
        print(f"metrics: {result['metrics_path']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
