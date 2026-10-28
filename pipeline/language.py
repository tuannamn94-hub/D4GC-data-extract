"""Language detection.

Prefers ``langdetect`` when installed. Falls back to a compact built-in
detector based on stop-word and script heuristics so the pipeline works
offline. Returns an ISO 639-1 code (e.g. "en", "es", "fr") or ``None``.
"""

from __future__ import annotations

import re

try:  # optional dependency
    from langdetect import detect as _ld_detect  # type: ignore
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0  # deterministic output
    _HAVE_LANGDETECT = True
except Exception:  # pragma: no cover
    _HAVE_LANGDETECT = False


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Small stop-word sets for a lightweight fallback detector.
_STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "is", "in", "to", "of", "a", "that", "it", "for", "on", "with", "as", "are"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "las", "un", "una", "por", "con", "para", "es"},
    "fr": {"le", "la", "les", "de", "et", "des", "un", "une", "que", "est", "pour", "dans", "sur", "pas"},
    "de": {"der", "die", "das", "und", "ist", "den", "von", "zu", "mit", "ein", "eine", "nicht", "auch"},
    "it": {"il", "la", "di", "che", "e", "un", "una", "per", "con", "non", "sono", "come", "del"},
    "pt": {"o", "a", "de", "que", "e", "do", "da", "em", "um", "uma", "para", "com", "nao", "os"},
}


def detect_language(text: str) -> str | None:
    """Detect the dominant language of ``text``."""
    sample = text.strip()
    if len(sample) < 3:
        return None

    script = _detect_by_script(sample)
    if script:
        return script

    if _HAVE_LANGDETECT:
        try:
            return _ld_detect(sample)
        except Exception:
            pass

    return _detect_by_stopwords(sample)


def _detect_by_script(text: str) -> str | None:
    """Detect languages with a distinctive script via Unicode code points."""
    counts = {"zh": 0, "ja": 0, "ko": 0, "ru": 0, "ar": 0, "el": 0, "he": 0}
    for ch in text[:2000]:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            counts["zh"] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts["ja"] += 1
        elif 0xAC00 <= o <= 0xD7A3:
            counts["ko"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["ru"] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts["ar"] += 1
        elif 0x0370 <= o <= 0x03FF:
            counts["el"] += 1
        elif 0x0590 <= o <= 0x05FF:
            counts["he"] += 1
    best = max(counts, key=counts.get)
    letters = sum(1 for ch in text[:2000] if ch.isalpha())
    if letters and counts[best] / letters > 0.2:
        return best
    return None


def _detect_by_stopwords(text: str) -> str | None:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if not tokens:
        return None
    token_set = set(tokens)
    scores = {
        lang: len(token_set & words) + sum(1 for t in tokens if t in words) * 0.1
        for lang, words in _STOPWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"
