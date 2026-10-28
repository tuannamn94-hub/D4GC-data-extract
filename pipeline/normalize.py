"""Text normalization and validation."""

from __future__ import annotations

import re
import unicodedata

# Zero-width and other invisible characters to strip.
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
# Collapse runs of horizontal whitespace.
_HSPACE_RE = re.compile(r"[ \t\f\v]+")
# Collapse 3+ blank lines into a single blank line.
_MULTI_NEWLINE_RE = re.compile(r"\n\s*\n\s*\n+")
# Control characters except tab/newline/carriage-return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Return a cleaned, canonical version of ``text``.

    Applies Unicode NFC normalization, removes invisible/control characters,
    normalizes newlines, and collapses redundant whitespace while preserving
    paragraph breaks.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    # Normalize per line, then collapse excess blank lines.
    lines = [_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


class ValidationError(ValueError):
    """Raised when extracted text fails validation."""


def validate_text(text: str, *, min_chars: int = 1, min_printable_ratio: float = 0.6) -> None:
    """Validate that ``text`` looks like usable content.

    Raises :class:`ValidationError` if the text is empty, too short, or made up
    mostly of non-printable characters (a common sign of a failed extraction).
    """
    if text is None:
        raise ValidationError("text is None")
    stripped = text.strip()
    if len(stripped) < min_chars:
        raise ValidationError(f"text too short ({len(stripped)} < {min_chars} chars)")

    printable = sum(1 for ch in stripped if ch.isprintable() or ch in "\n\t")
    ratio = printable / max(len(stripped), 1)
    if ratio < min_printable_ratio:
        raise ValidationError(
            f"printable ratio {ratio:.2f} below threshold {min_printable_ratio}"
        )
