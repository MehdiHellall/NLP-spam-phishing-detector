"""Text-level duplicate and conflict auditing."""

from __future__ import annotations

import pandas as pd

from email_threat_detector.constants import LABEL_COLUMN, TEXT_COLUMN
from email_threat_detector.data import validate_message_columns
from email_threat_detector.preprocessing import basic_text_preprocessor

TEXT_KEY_COLUMN = "_text_key"


def text_key(value: object) -> str:
    """Return the normalized key used to detect duplicate messages."""
    return basic_text_preprocessor(value)


def with_text_key(frame: pd.DataFrame, *, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    """Return a copy of a dataframe with a normalized text-key column."""
    return frame.assign(**{TEXT_KEY_COLUMN: frame[text_column].map(text_key)})


def audit_duplicate_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit repeated normalized text and conflicting labels."""
    validate_message_columns(frame)

    keyed = with_text_key(frame)
    grouped = (
        keyed.groupby(TEXT_KEY_COLUMN, sort=True)
        .agg(
            row_count=(TEXT_COLUMN, "size"),
            label_count=(LABEL_COLUMN, "nunique"),
            labels=(LABEL_COLUMN, lambda values: tuple(sorted(set(values)))),
            example_text=(TEXT_COLUMN, "first"),
        )
        .reset_index()
    )
    duplicates = grouped.loc[(grouped["row_count"] > 1) | (grouped["label_count"] > 1)].copy()
    return duplicates.assign(is_conflict=duplicates["label_count"] > 1).reset_index(drop=True)


def find_conflicting_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate-text rows where the same text has multiple labels."""
    audit = audit_duplicate_texts(frame)
    return audit.loc[audit["is_conflict"]].reset_index(drop=True)


def remove_conflicting_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove every row whose normalized text maps to multiple labels."""
    validate_message_columns(frame)

    conflicts = set(find_conflicting_texts(frame)[TEXT_KEY_COLUMN])
    if not conflicts:
        return frame.reset_index(drop=True).copy()

    keyed = with_text_key(frame)
    retained = keyed.loc[~keyed[TEXT_KEY_COLUMN].isin(conflicts)]
    return retained.drop(columns=[TEXT_KEY_COLUMN]).reset_index(drop=True)


def deduplicate_texts(frame: pd.DataFrame, *, keep: str = "first") -> pd.DataFrame:
    """Drop repeated normalized texts after conflicts have been handled."""
    validate_message_columns(frame)

    keyed = with_text_key(frame)
    deduplicated = keyed.drop_duplicates(subset=[TEXT_KEY_COLUMN], keep=keep)
    return deduplicated.drop(columns=[TEXT_KEY_COLUMN]).reset_index(drop=True)


def prepare_modeling_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove label conflicts and duplicate text before model splitting."""
    conflict_free = remove_conflicting_texts(frame)
    return deduplicate_texts(conflict_free)
