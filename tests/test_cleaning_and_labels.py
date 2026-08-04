import pandas as pd
import pytest

from email_threat_detector.cleaning import clean_messages, drop_null_and_empty_texts
from email_threat_detector.constants import labels_to_ids, normalize_label
from email_threat_detector.data import load_messages_csv, validate_message_columns
from email_threat_detector.preprocessing import basic_text_preprocessor, has_text, normalize_text


def test_normalize_label_maps_aliases_case_and_numeric_values() -> None:
    assert normalize_label(" PHISHING ") == "phish"
    assert normalize_label("Junk") == "spam"
    assert normalize_label("legit") == "ham"
    assert normalize_label(0) == "ham"
    assert normalize_label(2.0) == "spam"
    assert labels_to_ids(["normal", "scam", 2]) == [0, 1, 2]


@pytest.mark.parametrize("bad_label", [None, "malware", 99, 3.5])
def test_normalize_label_rejects_unknown_values(bad_label: object) -> None:
    with pytest.raises(ValueError):
        normalize_label(bad_label)


def test_normalize_text_decodes_html_collapses_whitespace_and_null_bytes() -> None:
    raw_text = "  FREE&nbsp;\n\tMoney\x00 now  "

    assert normalize_text(None) == ""
    assert normalize_text(raw_text) == "FREE Money now"
    assert basic_text_preprocessor(raw_text) == "free money now"
    assert has_text(raw_text)
    assert not has_text(" \n\t ")
    assert normalize_text(float("nan")) == ""
    assert normalize_text(pd.NA) == ""


def test_validate_message_columns_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="label"):
        validate_message_columns(pd.DataFrame({"text": ["hello"]}))


def test_drop_null_and_empty_texts_removes_blank_text_and_null_labels() -> None:
    frame = pd.DataFrame(
        {
            "text": ["usable message", "   ", "missing label"],
            "label": ["ham", "spam", None],
        }
    )

    cleaned = drop_null_and_empty_texts(frame)

    assert cleaned.to_dict("records") == [{"text": "usable message", "label": "ham"}]


def test_clean_messages_standardizes_text_labels_and_exact_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "text": [
                "  Free&nbsp; MONEY\x00 now ",
                "Free MONEY now",
                "\n\t",
                None,
                "Account looks normal",
                "Missing label",
            ],
            "label": ["junk", "spam", "ham", "spam", "0", None],
        }
    )

    cleaned = clean_messages(frame)

    assert cleaned.to_dict("records") == [
        {"text": "Free MONEY now", "label": "spam"},
        {"text": "Account looks normal", "label": "ham"},
    ]


def test_load_csv_allows_blank_labels_for_cleanup(tmp_path) -> None:
    csv_path = tmp_path / "messages.csv"
    csv_path.write_text(
        "text,label\n"
        "\"Project update\",0\n"
        "\"No label here\",\n"
        "\"Free coupon\",spam\n",
        encoding="utf-8",
    )

    loaded = load_messages_csv(csv_path)
    cleaned = clean_messages(loaded)

    assert cleaned.to_dict("records") == [
        {"text": "Project update", "label": "ham"},
        {"text": "Free coupon", "label": "spam"},
    ]
