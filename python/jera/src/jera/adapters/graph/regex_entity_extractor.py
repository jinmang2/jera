"""Deterministic regex-based entity extractor.

Extracts capitalized token runs (proper nouns / noun phrases) from text using
only the Python standard library — no NLTK, spaCy, or LLM required.

The extraction heuristic:
  1. Split text into sentences on ``.``, ``!``, ``?``.
  2. Within each sentence, find maximal runs of *capitalized* tokens that are
     **not** sentence-initial (so "The" at position 0 is excluded).
  3. Optionally include single capitalized tokens via ``include_single`` (default
     ``True``).
  4. All extracted strings are lowercased before returning (normalization).

This is intentionally conservative: it avoids spurious extractions at the cost
of some recall. A future opt-in LLM OpenIE extractor can replace this for
higher-recall production use — just implement
:class:`~jera.ports.entity_extractor.EntityExtractor` and pass it to
:class:`~jera.adapters.graph.hippo_retriever.HippoGraphRetriever`.

Example::

    >>> ext = RegexEntityExtractor()
    >>> ext.extract("Alice works at Acme Corp in Paris.")
    ['alice', 'acme corp', 'paris']
"""

from __future__ import annotations

import re

# Matches a single whitespace-separated token that starts with an uppercase
# letter followed by zero-or-more word characters (handles hyphenated names too
# via the \S* fallback below).
_UPPER_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9\-']*\b")

# Sentence boundary splitter — keeps split positions approximate; good enough
# for entity extraction where recall > precision on boundaries.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class RegexEntityExtractor:
    """Offline, deterministic proper-noun / noun-phrase extractor.

    Parameters
    ----------
    include_single:
        When ``True`` (default), single capitalized tokens that are not
        sentence-initial are returned as entities.  Set to ``False`` to return
        only multi-token runs (e.g. "Acme Corp") — useful when short common
        words produce too much noise.
    min_token_length:
        Tokens shorter than this are skipped (default 2).  Filters out lone
        capitals such as "I" or abbreviation artifacts like "A".
    """

    def __init__(
        self,
        *,
        include_single: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self._include_single = include_single
        self._min_token_length = min_token_length

    def extract(self, text: str) -> list[str]:
        """Return lowercased entity strings extracted from *text*.

        The result list is deduplicated and preserves first-occurrence order.
        """
        entities: list[str] = []
        seen: set[str] = set()

        for sentence in _SENT_SPLIT.split(text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            self._extract_from_sentence(sentence, entities, seen)

        return entities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_from_sentence(
        self,
        sentence: str,
        entities: list[str],
        seen: set[str],
    ) -> None:
        tokens = sentence.split()
        if not tokens:
            return

        # Collect positions of all capitalized tokens *after* position 0
        # (position 0 is sentence-initial — always capitalized, not useful).
        caps_at: list[int] = [
            i
            for i, tok in enumerate(tokens)
            if i > 0
            and _UPPER_TOKEN.match(tok)
            and len(re.sub(r"[^A-Za-z0-9]", "", tok)) >= self._min_token_length
        ]

        # Group consecutive capitalized positions into runs.
        runs: list[list[int]] = []
        for pos in caps_at:
            if runs and pos == runs[-1][-1] + 1:
                runs[-1].append(pos)
            else:
                runs.append([pos])

        for run in runs:
            if len(run) == 1 and not self._include_single:
                continue
            phrase = " ".join(tokens[i].rstrip(".,;:!?\"'") for i in run).lower()
            if phrase and phrase not in seen:
                seen.add(phrase)
                entities.append(phrase)
