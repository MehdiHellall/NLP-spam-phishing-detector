"""Threat label helpers used by the API."""

from __future__ import annotations

LABEL_TO_ID = {
    "ham": 0,
    "phish": 1,
    "spam": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_NAMES = tuple(LABEL_TO_ID.keys())

LABEL_ALIASES = {
    "0": "ham",
    "1": "phish",
    "2": "spam",
    "legit": "ham",
    "legitimate": "ham",
    "normal": "ham",
    "phishing": "phish",
    "scam": "phish",
    "junk": "spam",
}


def normalize_label(value: object) -> str:
    """Return the canonical label name."""
    if value is None:
        raise ValueError("Label cannot be None.")

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    if isinstance(value, int):
        try:
            return ID_TO_LABEL[value]
        except KeyError as exc:
            raise ValueError(f"Unknown numeric label: {value!r}") from exc

    normalized = str(value).strip().casefold()
    if normalized in LABEL_TO_ID:
        return normalized
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]

    raise ValueError(f"Unknown label: {value!r}")
