"""Deterministic sampling helpers."""

from __future__ import annotations

import pandas as pd

from email_threat_detector.constants import DEFAULT_RANDOM_SEED, LABEL_COLUMN
from email_threat_detector.data import validate_message_columns


def balanced_sample(
    frame: pd.DataFrame,
    *,
    samples_per_class: int | None = None,
    label_column: str = LABEL_COLUMN,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Return a deterministic class-balanced sample."""
    validate_message_columns(frame, label_column=label_column)

    counts = frame[label_column].value_counts()
    if counts.empty:
        raise ValueError("Cannot sample from an empty dataframe.")

    target_count = int(counts.min()) if samples_per_class is None else samples_per_class
    if target_count <= 0:
        raise ValueError("samples_per_class must be positive.")
    if (counts < target_count).any():
        raise ValueError("samples_per_class exceeds at least one class count.")

    sampled = [
        group.sample(n=target_count, random_state=random_state)
        for _, group in frame.groupby(label_column, sort=True)
    ]
    return (
        pd.concat(sampled, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
