"""BridgeFollowupController — deterministic, heuristic follow-up query controller.

Implements the :class:`~jera.ports.followup_controller.FollowupController` protocol
using a pure-Python, offline-deterministic bridge-following heuristic.  No LLM is
required; it is safe for CI and offline tests.

Algorithm
---------
On each call the controller decides whether to STOP or to issue a follow-up query:

1. **STOP conditions (return None)**

   a. ``round_index >= max_hops - 1``  — hard hop cap reached.  The pipeline's own
      ``max_hops`` guard provides a second, independent safety net.
   b. *Coverage threshold*: the fraction of the original query's content tokens
      that appear in at least one accumulated chunk's text exceeds
      ``coverage_threshold`` (default 0.85).  When the corpus already contains the
      answer, a further hop is unlikely to help.

2. **Bridge query construction (return str)**

   a. Tokenise every accumulated chunk's text into lower-case word tokens.
   b. Collect all tokens that are *not* already present in the running query
      vocabulary (i.e. not in the original query OR any prior follow-up query).
      These are the *new entities/terms* introduced by the retrieved evidence —
      the "bridge" in bridge-entity multi-hop reasoning (A-RAG §3, IRCoT §4).
   c. Rank the new tokens by document frequency (how many distinct chunks they
      appear in) — more widely attested bridging terms are more likely to be
      important.
   d. Take the top ``max_bridge_terms`` tokens (default 6) and append them to the
      original query, producing the next retrieval query.
      Example:
        original: "who founded the company that makes iPhone"
        hop-0 chunks introduce: "apple", "inc", "steve", "jobs", "cupertino"
        next query: "who founded the company that makes iPhone apple inc steve jobs"

Heuristic rationale
-------------------
This is a *pseudo-relevance / vocabulary expansion* step akin to Rocchio (1971)
but applied to bridge-entity reasoning rather than term frequency weighting.  The
key insight (from IRCoT and Search-R1) is that the bridge entity — a term that
connects the original query to the final answer — is often introduced in the first
retrieval round and is absent from the original query.  Appending it drives the
next retrieval toward the answer chunk.

LLM controller (future opt-in)
-------------------------------
A stronger controller would call an LLM with the original query and accumulated
chunks to *reason* about what is still unknown, then emit a targeted sub-question.
That approach (the true IRCoT / Search-R1 "think" step) should implement the same
``FollowupController`` protocol and be constructed only when ``enable_cloud=True``
with an appropriate API key — see the pattern in ``claude_hypothesis_llm.py`` and
``config/registry.py::_build_query_transformer``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jera.domain.chunk import Chunk

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


class BridgeFollowupController:
    """Deterministic bridge-following follow-up controller (no LLM).

    Parameters
    ----------
    max_hops:
        Maximum number of retrieval rounds (inclusive).  The controller returns
        ``None`` once ``round_index >= max_hops - 1`` so the loop never exceeds
        this count even if the pipeline's own ``max_hops`` guard is relaxed.
        Must be >= 1.
    coverage_threshold:
        Fraction of the original query's content tokens that must be covered by
        accumulated chunks to trigger an early STOP.  Default 0.85 (85 %).
        Set to 1.0 to disable coverage-based early stopping.
    max_bridge_terms:
        Maximum number of new bridge tokens to append to the next query.
        Default 6.
    min_token_length:
        Bridge tokens shorter than this are ignored (filters noise stopwords like
        "a", "in", "of").  Default 2.

    Bridge-term extraction scope
    ----------------------------
    Bridge terms are extracted only from chunks that are **new in the most recent
    round** (i.e. not present in the previous call's accumulated set).  Scanning
    all accumulated chunks would include distractor chunks gathered in earlier
    rounds, whose tokens would dilute the ranking and potentially displace the
    genuine bridge entity.  Focusing on newly retrieved chunks mirrors the IRCoT
    reasoning step: "given what I just found, what should I look for next?"
    """

    def __init__(
        self,
        *,
        max_hops: int = 3,
        coverage_threshold: float = 0.85,
        max_bridge_terms: int = 6,
        min_token_length: int = 2,
    ) -> None:
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        self._max_hops = max_hops
        self._coverage_threshold = coverage_threshold
        self._max_bridge_terms = max_bridge_terms
        self._min_token_len = min_token_length
        # Tracks the union of all query vocabularies issued so far.
        self._issued_vocab: set[str] = set()
        # chunk_ids seen at the end of the previous round (to isolate new chunks).
        self._prev_chunk_ids: set[str] = set()

    def next_query(
        self,
        original_query: str,
        accumulated_chunks: Sequence[Chunk],
        round_index: int,
    ) -> str | None:
        """Return the next retrieval query, or None to stop the loop.

        Called once per completed round.  ``round_index`` is 0-based (0 = after
        the first retrieval round).

        Side effect
        -----------
        On the first call (``round_index == 0``) the original query's vocabulary
        is seeded into ``_issued_vocab``.  On subsequent calls the returned query's
        vocabulary is merged in.  This state is reset automatically between
        ``IterativeRetrievalPipeline.answer`` calls (the pipeline constructs a
        fresh controller per call — or the caller must reset state manually).

        Reset
        -----
        If you reuse a single ``BridgeFollowupController`` instance across multiple
        ``IterativeRetrievalPipeline.answer`` calls, call ``reset()`` between them
        to clear the issued-vocabulary state.
        """
        # Seed issued vocab on first round.
        if round_index == 0:
            self._issued_vocab = _token_set(original_query)
            self._prev_chunk_ids = set()

        # --- STOP condition (a): hard hop cap ---
        if round_index >= self._max_hops - 1:
            return None

        # --- STOP condition (b): coverage threshold ---
        query_tokens = _token_set(original_query)
        # Filter to "content" tokens (length >= min_token_length).
        content_tokens = {t for t in query_tokens if len(t) >= self._min_token_len}
        if content_tokens:
            chunk_text_union = " ".join(c.text for c in accumulated_chunks)
            chunk_tokens = _token_set(chunk_text_union)
            covered = content_tokens & chunk_tokens
            coverage = len(covered) / len(content_tokens)
            if coverage >= self._coverage_threshold:
                return None

        if not accumulated_chunks:
            return None

        # --- Bridge query construction ---
        # Only consider chunks that are *new* since the previous round.
        # This prevents distractor tokens from earlier rounds polluting the bridge
        # term ranking and displacing the genuine bridge entity.
        new_chunks = [c for c in accumulated_chunks if c.chunk_id not in self._prev_chunk_ids]

        # Update prev_chunk_ids for next round.
        self._prev_chunk_ids = {c.chunk_id for c in accumulated_chunks}

        if not new_chunks:
            return None

        # Count document frequency of each new token across *new* chunks only.
        df: dict[str, int] = {}
        for chunk in new_chunks:
            chunk_tokens_set = _token_set(chunk.text)
            for token in chunk_tokens_set:
                if token not in self._issued_vocab and len(token) >= self._min_token_len:
                    df[token] = df.get(token, 0) + 1

        if not df:
            # No new bridging terms found — further retrieval is unlikely to help.
            return None

        # Rank by document frequency descending, break ties lexicographically.
        ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        bridge_terms = [t for t, _ in ranked[: self._max_bridge_terms]]

        # Build the next query by appending bridge terms to the original.
        next_q = original_query + " " + " ".join(bridge_terms)

        # Track vocabulary so subsequent rounds don't re-use these terms as bridges.
        self._issued_vocab.update(bridge_terms)

        return next_q

    def reset(self) -> None:
        """Clear issued-vocabulary state so this instance can be reused across calls."""
        self._issued_vocab = set()
        self._prev_chunk_ids = set()
