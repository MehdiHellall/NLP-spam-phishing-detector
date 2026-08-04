"""Data loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from email_threat_detector.constants import LABEL_COLUMN, TEXT_COLUMN


def validate_message_columns(
    frame: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> None:
    """Validate that a dataframe contains the expected message columns."""
    missing = {text_column, label_column}.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_names}")


def load_messages_csv(
    path: str | Path,
    *,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Load a CSV message dataset and return canonical text/label columns.

    Label normalization happens during cleaning so rows with blank labels can
    be removed before mapping.
    """
    frame = pd.read_csv(path)
    validate_message_columns(frame, text_column=text_column, label_column=label_column)

    selected = frame.loc[:, [text_column, label_column]].rename(
        columns={text_column: TEXT_COLUMN, label_column: LABEL_COLUMN}
    )
    return selected
