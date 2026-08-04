"""Deterministic stratified splitting and leakage checks."""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.model_selection import train_test_split

from email_threat_detector.constants import DEFAULT_RANDOM_SEED, LABEL_COLUMN, TEXT_COLUMN
from email_threat_detector.data import validate_message_columns
from email_threat_detector.preprocessing import basic_text_preprocessor


def stratified_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    label_column: str = LABEL_COLUMN,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Split a dataframe into deterministic train/validation/test dataframes."""
    validate_message_columns(frame, label_column=label_column)
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be in [0, 1).")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[label_column],
    )

    if validation_size == 0:
        return {
            "train": train_validation.reset_index(drop=True),
            "validation": frame.iloc[0:0].copy(),
            "test": test.reset_index(drop=True),
        }

    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_validation[label_column],
    )

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def normalized_text_set(frame: pd.DataFrame, *, text_column: str = TEXT_COLUMN) -> set[str]:
    """Return normalized text values for overlap checks."""
    return set(frame[text_column].map(basic_text_preprocessor))


def find_text_overlaps(splits: dict[str, pd.DataFrame]) -> dict[tuple[str, str], set[str]]:
    """Find normalized text overlaps between split dataframes."""
    overlaps: dict[tuple[str, str], set[str]] = {}
    for left_name, right_name in combinations(sorted(splits), 2):
        overlap = normalized_text_set(splits[left_name]) & normalized_text_set(splits[right_name])
        if overlap:
            overlaps[(left_name, right_name)] = overlap
    return overlaps


def assert_no_text_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Raise when duplicate text crosses split boundaries."""
    overlaps = find_text_overlaps(splits)
    if overlaps:
        pairs = ", ".join(
            f"{left}/{right}: {len(values)}" for (left, right), values in overlaps.items()
        )
        raise ValueError(f"Text leakage across splits detected: {pairs}")
