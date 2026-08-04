"""Text normalization used by data cleaning and sklearn vectorizers."""

from __future__ import annotations

import html
import math
import re
import unicodedata
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Normalize message text without removing potentially useful tokens."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if _is_missing_scalar(value):
        return ""

    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _is_missing_scalar(value: Any) -> bool:
    """Return True for numpy/pandas-style missing scalars."""
    try:
        import pandas as pd
    except ImportError:
        return False

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def basic_text_preprocessor(value: object) -> str:
    """Normalize and lowercase text for vectorizers."""
    return normalize_text(value).casefold()


def has_text(value: object) -> bool:
    """Return True when a value contains non-whitespace text."""
    return bool(normalize_text(value))
