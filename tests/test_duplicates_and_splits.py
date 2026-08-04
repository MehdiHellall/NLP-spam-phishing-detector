import pandas as pd
import pytest

from email_threat_detector.duplicates import (
    TEXT_KEY_COLUMN,
    audit_duplicate_texts,
    find_conflicting_texts,
    prepare_modeling_data,
    remove_conflicting_texts,
)
from email_threat_detector.sampling import balanced_sample
from email_threat_detector.splits import (
    assert_no_text_overlap,
    find_text_overlaps,
    stratified_train_validation_test_split,
)


def _duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "Win cash now",
                " win   cash now ",
                "Project update",
                "project UPDATE",
                "Unique notice",
            ],
            "label": ["spam", "spam", "ham", "phish", "ham"],
        }
    )


def test_duplicate_audit_flags_repeated_texts_and_label_conflicts() -> None:
    audit = audit_duplicate_texts(_duplicate_frame()).set_index(TEXT_KEY_COLUMN)

    assert set(audit.index) == {"project update", "win cash now"}
    assert audit.loc["win cash now", "row_count"] == 2
    assert audit.loc["win cash now", "label_count"] == 1
    assert not bool(audit.loc["win cash now", "is_conflict"])
    assert audit.loc["project update", "labels"] == ("ham", "phish")
    assert bool(audit.loc["project update", "is_conflict"])


def test_conflict_removal_and_deduplication_prepare_modeling_rows() -> None:
    frame = _duplicate_frame()

    conflicts = find_conflicting_texts(frame)
    conflict_free = remove_conflicting_texts(frame)
    prepared = prepare_modeling_data(frame)

    assert conflicts[TEXT_KEY_COLUMN].tolist() == ["project update"]
    assert conflict_free["text"].tolist() == [
        "Win cash now",
        " win   cash now ",
        "Unique notice",
    ]
    assert prepared.to_dict("records") == [
        {"text": "Win cash now", "label": "spam"},
        {"text": "Unique notice", "label": "ham"},
    ]


def test_conflict_free_duplicate_helpers_return_copies() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Status update", "Another note"],
            "label": ["ham", "ham"],
        }
    )

    assert find_conflicting_texts(frame).empty
    assert remove_conflicting_texts(frame).equals(frame.reset_index(drop=True))


def _balanced_split_frame() -> pd.DataFrame:
    rows = [
        {"text": f"{label} synthetic message {index}", "label": label}
        for label in ("ham", "phish", "spam")
        for index in range(10)
    ]
    return pd.DataFrame(rows)


def test_stratified_split_is_deterministic_and_preserves_label_balance() -> None:
    frame = _balanced_split_frame()

    first = stratified_train_validation_test_split(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_state=123,
    )
    second = stratified_train_validation_test_split(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_state=123,
    )

    assert {name: len(split) for name, split in first.items()} == {
        "train": 18,
        "validation": 6,
        "test": 6,
    }
    for split_name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first[split_name], second[split_name])

    assert first["train"]["label"].value_counts().sort_index().to_dict() == {
        "ham": 6,
        "phish": 6,
        "spam": 6,
    }
    assert first["validation"]["label"].value_counts().sort_index().to_dict() == {
        "ham": 2,
        "phish": 2,
        "spam": 2,
    }
    assert first["test"]["label"].value_counts().sort_index().to_dict() == {
        "ham": 2,
        "phish": 2,
        "spam": 2,
    }


def test_split_validation_zero_and_invalid_sizes() -> None:
    frame = _balanced_split_frame()

    splits = stratified_train_validation_test_split(
        frame,
        validation_size=0,
        test_size=0.2,
        random_state=123,
    )

    assert splits["validation"].empty
    with pytest.raises(ValueError, match="validation_size"):
        stratified_train_validation_test_split(frame, validation_size=-0.1)
    with pytest.raises(ValueError, match="test_size"):
        stratified_train_validation_test_split(frame, test_size=0)
    with pytest.raises(ValueError, match="less than 1"):
        stratified_train_validation_test_split(frame, validation_size=0.6, test_size=0.5)


def test_text_overlap_detection_normalizes_train_test_text() -> None:
    splits = {
        "train": pd.DataFrame(
            {"text": ["Win CASH now", "team update"], "label": ["spam", "ham"]}
        ),
        "test": pd.DataFrame(
            {"text": [" win   cash now ", "reset password"], "label": ["spam", "phish"]}
        ),
        "validation": pd.DataFrame(
            {"text": ["invoice paid"], "label": ["ham"]}
        ),
    }

    overlaps = find_text_overlaps(splits)

    assert overlaps[("test", "train")] == {"win cash now"}
    with pytest.raises(ValueError, match="Text leakage across splits detected"):
        assert_no_text_overlap(splits)


def test_text_overlap_check_passes_for_disjoint_splits() -> None:
    splits = {
        "train": pd.DataFrame({"text": ["status report"], "label": ["ham"]}),
        "test": pd.DataFrame({"text": ["urgent reset"], "label": ["phish"]}),
    }

    assert find_text_overlaps(splits) == {}
    assert_no_text_overlap(splits)


def test_balanced_sample_is_deterministic_and_validates_requested_size() -> None:
    frame = pd.DataFrame(
        [
            {"text": f"{label} message {index}", "label": label}
            for label, count in {"ham": 4, "phish": 3, "spam": 5}.items()
            for index in range(count)
        ]
    )

    first = balanced_sample(frame, samples_per_class=2, random_state=9)
    second = balanced_sample(frame, samples_per_class=2, random_state=9)

    pd.testing.assert_frame_equal(first, second)
    assert first["label"].value_counts().sort_index().to_dict() == {
        "ham": 2,
        "phish": 2,
        "spam": 2,
    }

    with pytest.raises(ValueError, match="exceeds"):
        balanced_sample(frame, samples_per_class=6)

    with pytest.raises(ValueError, match="positive"):
        balanced_sample(frame, samples_per_class=0)

    with pytest.raises(ValueError, match="empty"):
        balanced_sample(frame.iloc[0:0])
