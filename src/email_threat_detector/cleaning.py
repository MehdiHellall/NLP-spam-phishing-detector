"""Null, empty-text, and exact-row cleanup for message data."""

from __future__ import annotations

import pandas as pd

from email_threat_detector.constants import LABEL_COLUMN, TEXT_COLUMN, normalize_label
from email_threat_detector.data import validate_message_columns
from email_threat_detector.preprocessing import normalize_text


def standardize_messages(
    frame: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Return canonical text/label columns with normalized values."""
    validate_message_columns(frame, text_column=text_column, label_column=label_column)

    selected = frame.loc[:, [text_column, label_column]].rename(
        columns={text_column: TEXT_COLUMN, label_column: LABEL_COLUMN}
    )
    selected = selected.dropna(subset=[LABEL_COLUMN]).copy()
    return selected.assign(
        text=selected[TEXT_COLUMN].map(normalize_text),
        label=selected[LABEL_COLUMN].map(normalize_label),
    )


def drop_null_and_empty_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without usable text or labels."""
    validate_message_columns(frame)

    cleaned = frame.dropna(subset=[LABEL_COLUMN]).copy()
    cleaned = cleaned.loc[cleaned[TEXT_COLUMN].map(lambda value: bool(normalize_text(value)))]
    return cleaned.reset_index(drop=True)


def clean_messages(
    frame: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Canonicalize labels/text, remove empty rows, and drop exact duplicates."""
    standardized = standardize_messages(
        frame,
        text_column=text_column,
        label_column=label_column,
    )
    non_empty = drop_null_and_empty_texts(standardized)
    return non_empty.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
