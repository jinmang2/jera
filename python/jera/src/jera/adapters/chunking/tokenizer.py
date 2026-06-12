"""Deterministic whitespace tokenizer used for chunk sizing.

Intentionally simple and dependency-free (no tiktoken) so chunking is reproducible and the
token_count provenance field is stable across environments. Documented as such in the ADR.
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    return len(text.split())
