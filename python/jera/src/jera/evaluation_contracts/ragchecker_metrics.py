"""RAGChecker-style claim-level evaluation metrics (pure, deterministic, no IO).

Implements five metric families inspired by:
- RAGChecker (Ru et al., arXiv:2408.08067, NeurIPS 2024): claim-level grounding diagnostics.
- TREC 2025 RAG Track (arXiv:2603.09891): weighted Full/Partial/None citation support.
- RGB (Chen et al., arXiv:2309.01431, AAAI 2024): negative-rejection / abstention scoring.

All functions are offline-deterministic: claims are approximated by sentence splitting
(reusing ``_sentences`` / ``_tokens`` / ``_containment`` from ``generation_metrics``),
avoiding any LLM or NLI model dependency while preserving the same contract structure
as the full RAGChecker pipeline.

Metrics exported:
- ``claim_precision``     — anti-hallucination at claim granularity (FActScore-style)
- ``claim_recall``        — completeness relative to a gold reference answer
- ``noise_sensitivity``   — fraction of answer claims induced by noisy (off-gold) context chunks
- ``citation_precision``  — weighted fraction of cited chunks that support their answer sentence
- ``citation_recall``     — fraction of answer sentences that have at least partial citation support
- ``abstention_score``    — 1.0 when the answer hedges / abstains (RGB negative rejection)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from jera.evaluation_contracts.generation_metrics import (
    _containment,
    _sentences,
    _tokens,
)

# ---------------------------------------------------------------------------
# Abstention lexicon (RGB — negative rejection)
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(
    r"(?:"
    # English hedge phrases (word-boundary anchored)
    r"\b(?:"
    r"i don'?t know"
    r"|i do not know"
    r"|cannot determine"
    r"|can'?t determine"
    r"|not enough information"
    r"|don'?t have enough"
    r"|do not have enough"
    r"|insufficient(?:\s+(?:context|information))?"
    r"|unable to (?:answer|determine|find)"
    r"|no information"
    r"|unclear from"
    r"|not specified"
    r"|no relevant"
    r"|cannot answer"
    r"|can'?t answer"
    r")\b"
    # Korean hedge phrases (no \b — CJK chars have no word boundary)
    r"|모르겠"
    r"|알 수 없"
    r"|정보가 없"
    r"|확인할 수 없"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# TREC 2025 weighted-support thresholds
# ---------------------------------------------------------------------------

_FULL_SUPPORT_THRESHOLD: float = 0.8  # containment ≥ 0.8 → Full support (1.0)
_PARTIAL_SUPPORT_THRESHOLD: float = 0.4  # 0.4 ≤ containment < 0.8 → Partial (0.5)
# containment < 0.4 → No support (0.0)


def _weighted_support(sentence_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Return TREC 2025 weighted support score: Full=1.0, Partial=0.5, None=0.0."""
    c = _containment(sentence_tokens, chunk_tokens)
    if c >= _FULL_SUPPORT_THRESHOLD:
        return 1.0
    if c >= _PARTIAL_SUPPORT_THRESHOLD:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# 1. claim_precision
# ---------------------------------------------------------------------------


def claim_precision(
    answer_text: str,
    context_texts: Sequence[str],
    *,
    threshold: float = 0.6,
) -> float:
    """Fraction of answer claims (sentences) grounded in the retrieved context.

    Measures anti-hallucination at claim granularity — the FActScore-style
    "precision" of the generated answer against its context evidence.  A claim is
    "supported" when ``_containment(claim_tokens, union_context_tokens) ≥ threshold``.

    Args:
        answer_text:    Generated answer (split into claims by sentence boundaries).
        context_texts:  Retrieved context chunks (union taken as the evidence pool).
        threshold:      Minimum token-containment ratio for a claim to be supported.

    Returns:
        Float in [0, 1].  1.0 = every claim is grounded; 0.0 = nothing grounded.
        Vacuously 1.0 when ``answer_text`` is empty.
    """
    claims = _sentences(answer_text)
    if not claims:
        return 1.0
    ctx_tokens: set[str] = set()
    for ctx in context_texts:
        ctx_tokens |= _tokens(ctx)
    if not ctx_tokens:
        return 0.0
    supported = sum(1 for c in claims if _containment(_tokens(c), ctx_tokens) >= threshold)
    return supported / len(claims)


# ---------------------------------------------------------------------------
# 2. claim_recall
# ---------------------------------------------------------------------------


def claim_recall(
    answer_text: str,
    gold_text: str,
    *,
    threshold: float = 0.6,
) -> float:
    """Fraction of gold-answer claims covered by the generated answer (completeness).

    A gold claim is "covered" when ``_containment(gold_claim_tokens, answer_tokens)
    ≥ threshold`` — i.e. most of the gold claim's tokens appear somewhere in the
    answer.  This is the recall side of FActScore / RAGChecker overall recall.

    Args:
        answer_text:  Generated answer text.
        gold_text:    Reference (gold) answer text.
        threshold:    Minimum containment ratio to count a gold claim as covered.

    Returns:
        Float in [0, 1].  1.0 = all gold claims reproduced; 0.0 = none covered.
        Vacuously 1.0 when ``gold_text`` is empty (nothing to recall).
    """
    gold_claims = _sentences(gold_text)
    if not gold_claims:
        return 1.0
    answer_tokens = _tokens(answer_text)
    if not answer_tokens:
        return 0.0
    covered = sum(1 for gc in gold_claims if _containment(_tokens(gc), answer_tokens) >= threshold)
    return covered / len(gold_claims)


# ---------------------------------------------------------------------------
# 3. noise_sensitivity
# ---------------------------------------------------------------------------


def noise_sensitivity(
    answer_text: str,
    context_texts: Sequence[str],
    gold_text: str,
    *,
    threshold: float = 0.6,
) -> float:
    """Fraction of answer claims induced by noisy (off-gold) context chunks.

    Diagnoses whether retrieval noise caused the generator to produce unfaithful
    content.  Implements the RAGChecker / RAGAS noise-sensitivity approximation:

    A "noisy chunk" is a context chunk whose own tokens are NOT grounded in the gold
    answer (``_containment(chunk_tokens, gold_tokens) < threshold``).

    An answer claim is "noise-induced" when **both**:

    1. The claim is NOT grounded in the gold answer
       (``_containment(claim_tokens, gold_tokens) < threshold``).
    2. The claim IS similar to at least one noisy chunk
       (``_containment(claim_tokens, noisy_chunk_tokens) ≥ threshold``).

    Returns 0.0 when the answer is fully faithful to gold or when no noisy chunks
    exist.  Returns > 0.0 when the generator drifted toward content from noisy chunks.

    Args:
        answer_text:    Generated answer.
        context_texts:  All retrieved context chunks (may include noisy ones).
        gold_text:      Reference (gold) answer — defines what counts as "correct."
        threshold:      Containment ratio cutoff used throughout.

    Returns:
        Float in [0, 1].  Higher = more noise-induced hallucination.
    """
    claims = _sentences(answer_text)
    if not claims:
        return 0.0
    gold_tokens = _tokens(gold_text)

    # Identify noisy chunks: those not grounded in gold
    noisy_chunks: list[set[str]] = []
    for ctx in context_texts:
        ctx_tok = _tokens(ctx)
        if not ctx_tok:
            continue
        if _containment(ctx_tok, gold_tokens) < threshold:
            noisy_chunks.append(ctx_tok)

    if not noisy_chunks:
        return 0.0

    noise_induced = 0
    for claim in claims:
        claim_tok = _tokens(claim)
        # Condition 1: claim is NOT grounded in gold
        if _containment(claim_tok, gold_tokens) >= threshold:
            continue
        # Condition 2: claim traces to a noisy chunk
        if any(_containment(claim_tok, nc) >= threshold for nc in noisy_chunks):
            noise_induced += 1

    return noise_induced / len(claims)


# ---------------------------------------------------------------------------
# 4. citation_precision & citation_recall (TREC 2025 weighted support)
# ---------------------------------------------------------------------------


def citation_precision(
    answer_sentences_to_cited_texts: Mapping[str, Sequence[str]],
) -> float:
    """Weighted fraction of citations that support their corresponding answer sentence.

    Implements TREC 2025 RAG Track weighted support: each (sentence, cited_chunk)
    pair is graded as Full support (1.0), Partial support (0.5), or No support (0.0)
    using token-containment thresholds (≥0.8 Full, ≥0.4 Partial, else None).

    The score per sentence is the **maximum** weighted support across its cited chunks
    (a sentence is well-supported if any one of its citations fully backs it).
    The overall metric is the mean across all sentences.

    Args:
        answer_sentences_to_cited_texts:
            Mapping from each answer sentence (str) to the list of chunk texts cited
            for that sentence.  Sentences with empty citation lists score 0.0.

    Returns:
        Float in [0, 1].  1.0 = every sentence has a fully-supporting citation.
        Returns 1.0 vacuously when the mapping is empty (no sentences to grade).

    Example::

        citation_precision({
            "Paris is the capital of France.": ["Paris is the capital of France."],
            "It has 12 million people.": [],  # uncited — scores 0.0
        })
        # → 0.5  (one fully supported, one unsupported)
    """
    if not answer_sentences_to_cited_texts:
        return 1.0
    total = 0.0
    for sent, cited_chunks in answer_sentences_to_cited_texts.items():
        sent_tok = _tokens(sent)
        if not cited_chunks:
            # No citation → no support
            total += 0.0
            continue
        best = max(_weighted_support(sent_tok, _tokens(chunk)) for chunk in cited_chunks)
        total += best
    return total / len(answer_sentences_to_cited_texts)


def citation_recall(
    answer_sentences_to_cited_texts: Mapping[str, Sequence[str]],
) -> float:
    """Fraction of answer sentences that have at least partial citation support.

    A sentence is "recalled" (cited) when its best weighted-support score across its
    cited chunks is > 0.0 (i.e., at least Partial support ≥ 0.4 containment).

    The signature mirrors ``citation_precision`` for API symmetry: both accept the
    same ``{sentence: [chunk_text, ...]}`` mapping, so callers can compute both from
    a single attribution pass.

    Args:
        answer_sentences_to_cited_texts:
            Mapping from each answer sentence to the list of chunk texts cited for it.

    Returns:
        Float in [0, 1].  1.0 = every sentence has a supporting citation.
        Returns 1.0 vacuously when the mapping is empty.

    Example::

        citation_recall({
            "Paris is the capital.": ["Paris is the capital of France."],
            "The moon is made of cheese.": [],
        })
        # → 0.5  (one cited, one not)
    """
    if not answer_sentences_to_cited_texts:
        return 1.0
    recalled = 0
    for sent, cited_chunks in answer_sentences_to_cited_texts.items():
        sent_tok = _tokens(sent)
        if not cited_chunks:
            continue
        best = max(_weighted_support(sent_tok, _tokens(chunk)) for chunk in cited_chunks)
        if best > 0.0:
            recalled += 1
    return recalled / len(answer_sentences_to_cited_texts)


# ---------------------------------------------------------------------------
# 5. abstention_score (RGB — negative rejection)
# ---------------------------------------------------------------------------


def abstention_score(answer_text: str) -> float:
    """Return 1.0 if the answer hedges or abstains, else 0.0.

    Deterministic CI stand-in for the RGB benchmark's "Negative Rejection" axis
    (Chen et al., AAAI 2024).  An appropriate abstention when context is absent
    should score 1.0; a confident unfounded assertion should score 0.0.

    Detection is lexical: a curated set of English and Korean hedge phrases is
    matched case-insensitively.  This provides a reliable signal for the most
    common abstention patterns without requiring an LLM judge.

    Hedge phrases include: "i don't know", "cannot determine", "no information",
    "insufficient context", "unable to answer", "unclear from", "not specified",
    "모르겠", "알 수 없", "정보가 없", "확인할 수 없", and variants.

    Args:
        answer_text: The generated answer string.

    Returns:
        1.0 if a hedge phrase is detected, 0.0 otherwise.
    """
    return 1.0 if _HEDGE_RE.search(answer_text) else 0.0
