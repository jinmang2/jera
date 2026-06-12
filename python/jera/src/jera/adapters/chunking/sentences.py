"""Deterministic sentence segmentation with character offsets.

Dependency-free (no nltk/spacy) so semantic chunking stays reproducible. Splits on sentence
terminators followed by whitespace, and on blank lines, preserving exact char spans.
"""

from __future__ import annotations

import re

# A sentence boundary: ., !, ? (optionally followed by quotes/brackets) then whitespace,
# OR a newline break.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]?\s+|\n+")


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return ``(sentence, start, end)`` triples covering the non-whitespace spans of ``text``."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start()
        segment = text[pos:end]
        stripped = segment.strip()
        if stripped:
            lead = len(segment) - len(segment.lstrip())
            start = pos + lead
            out.append((stripped, start, start + len(stripped)))
        pos = match.end()
    # trailing remainder
    segment = text[pos:]
    stripped = segment.strip()
    if stripped:
        lead = len(segment) - len(segment.lstrip())
        start = pos + lead
        out.append((stripped, start, start + len(stripped)))
    return out
