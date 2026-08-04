"""Text preprocessing used by the packaged sklearn artifact."""

from __future__ import annotations

import html
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def basic_text_preprocessor(value: object) -> str:
    return normalize_text(value).casefold()
