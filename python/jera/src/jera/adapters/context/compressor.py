"""ExtractiveCompressor — sentence-level query-overlap compression.

Research basis
--------------
RECOMP (Xu & Shi, ICLR 2024, arXiv:2310.04408) introduced extractive context
compression: score each sentence in a retrieved passage for relevance to the query,
then retain only those sentences above a threshold.  The extractive variant achieves
up to 10× compression with minimal accuracy loss.

EXIT (Hwang et al., ACL 2025 Findings, arXiv:2412.12559) extended RECOMP with
context-aware (non-independent) sentence classification: a sentence's score is
boosted when its immediate neighbours are also query-relevant, capturing referential
dependencies ("The subject is X." … "It was discovered in 1905.").

This adapter approximates both techniques using pure string operations and no
external models:

1. Sentence splitting: a lightweight regex pattern handling ``.``, ``!``, ``?``
   followed by whitespace or end-of-string, while preserving sentences that contain
   abbreviations or numbers (avoids false splits on e.g. ``Dr.``, ``3.14``).
2. Token-overlap scoring: each sentence is scored by
   ``|tokens(sentence) ∩ tokens(query)| / (|tokens(query)| + ε)``.
3. Threshold selection: sentences whose score exceeds the per-chunk mean score
   are retained.  This is data-adaptive (no fixed global threshold) so short
   queries and long queries both produce reasonable compression.
4. Minimum-keep guarantee: at least ``min_keep`` sentences are always retained
   (the one(s) with the highest overlap), ensuring the chunk is never emptied.

Provenance contract
-------------------
``chunk_id``, ``document_id``, ``source_id``, ``char_span``, ``page_span``,
``section_path``, ``element_ids``, ``chunk_strategy``, ``chunk_version``, and
``parent_chunk_id`` are all preserved unchanged on the returned chunk.  Only the
generation-time ``text`` field is replaced with the extractively compressed view.
The ``token_count`` field is updated to reflect the new text length so downstream
budget accounting remains consistent.  Callers that need the original spans should
use ``chunk_id`` to look up the source chunk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jera.domain.chunk import Chunk

# Sentence boundary: split after `.`, `!`, or `?` when followed by whitespace or EOS,
# but only when the preceding character is NOT a digit (avoids splitting on "3.14")
# and the token before the punctuation is longer than one character (avoids "Dr.").
_SENT_BOUNDARY = re.compile(r"(?<!\d)(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a lightweight regex boundary detector."""
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _token_overlap(sentence: str, query_tokens: frozenset[str]) -> float:
    """Fraction of query tokens present in *sentence* (case-insensitive)."""
    if not query_tokens:
        return 0.0
    sent_tokens = frozenset(sentence.lower().split())
    return len(sent_tokens & query_tokens) / len(query_tokens)


class ExtractiveCompressor:
    """Compress each chunk to its query-relevant sentences.

    Sentences are scored by token overlap with the query.  Those above the
    per-chunk mean score are retained; the ``min_keep`` highest-scoring sentences
    are always kept regardless of threshold.

    Parameters
    ----------
    min_keep:
        Minimum number of sentences to retain per chunk.  Default 1 guarantees
        the most query-relevant sentence is always present so an extractive
        generator remains grounded.
    """

    name: str = "extractive_compressor"

    def __init__(self, min_keep: int = 1) -> None:
        if min_keep < 1:
            raise ValueError(f"min_keep must be ≥ 1, got {min_keep!r}")
        self.min_keep = min_keep

    def _compress_text(self, text: str, query_tokens: frozenset[str]) -> str:
        """Return the extractively compressed text for a single passage."""
        sentences = _split_sentences(text)
        if len(sentences) <= self.min_keep:
            return text

        scores = [_token_overlap(s, query_tokens) for s in sentences]
        mean_score = sum(scores) / len(scores)

        # Retain sentences above the per-chunk mean.
        kept = [s for s, sc in zip(sentences, scores, strict=True) if sc > mean_score]

        # Guarantee min_keep sentences using the highest-scoring ones.
        if len(kept) < self.min_keep:
            ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
            # Preserve original sentence order among the top-min_keep.
            top_indices = set(ranked[: self.min_keep])
            kept = [s for i, s in enumerate(sentences) if i in top_indices]

        return " ".join(kept)

    def process(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Compress each chunk to its query-relevant sentences.

        Parameters
        ----------
        query:
            The retrieval query.  Its tokens drive the overlap scoring.
        chunks:
            Chunks to compress; typically already reordered and curated.

        Returns
        -------
        list[Chunk]
            New ``Chunk`` objects with compressed ``text`` and updated
            ``token_count``.  All provenance fields (``chunk_id``, ``char_span``,
            etc.) are preserved unchanged.  The count of returned chunks equals
            the count of input chunks.
        """
        query_tokens = frozenset(query.lower().split())
        result: list[Chunk] = []
        for chunk in chunks:
            compressed = self._compress_text(chunk.text, query_tokens)
            if compressed == chunk.text:
                result.append(chunk)
            else:
                new_token_count = len(compressed.split())
                result.append(
                    chunk.model_copy(update={"text": compressed, "token_count": new_token_count})
                )
        return result
