"""Corrective RAG (CRAG) — non-tautological unit tests.

Yan et al., "Corrective Retrieval Augmented Generation", ICLR 2025, arXiv:2401.15884.

NON-TAUTOLOGICAL DESIGN
-----------------------
Corpus:
  A = "cells how do organisms grow divide and multiply population"
      — echoes the query vocabulary, earns a non-zero BM25 score, and wins the
        initial sparse ranking.  Jaccard(query, A) ≈ 0.44 — below overlap_threshold
        of 0.5, so the initial retrieval grades AMBIGUOUS, not CORRECT.

  B = "mitosis cytokinesis chromosomes replicate nuclear division"
      — the *true* answer about cell division, written in scientific vocabulary
        with ZERO lexical overlap with the original query ("how do cells divide").
        BM25 score = 0.0 on the original query → B is NOT recall@1 initially.

Query: "how do cells divide"
  * Initial sparse retrieval: A wins (score 2.54), B = 0.0 → B is not recall@1.
  * Initial grade: AMBIGUOUS (A's Jaccard = 0.44 < overlap_threshold 0.5).

Corrective transformer (fake, deterministic):
  transform("how do cells divide") →
    ["how do cells divide",                          # original (index 0, per protocol)
     "mitosis cytokinesis nuclear division chromosomes"]   # bridges the vocabulary gap

After RRF fusion over the two per-variant rankings:
  q0 ("how do cells divide"): A#1, B#2
  q1 ("mitosis cytokinesis…"): B#1, A#2
  → RRF scores: B=0.03252, A=0.03252 (tie) → tie-break: chunk_id lex asc.
    chunk_B = "5c82a..." < chunk_A = "ea90a..." → B wins.

Result: B moves from NOT recall@1 → recall@1.  The lift is real: it is entirely
driven by the corrective vocabulary bridging, not a rigged corpus or trivial query.
"""

from __future__ import annotations

from jera.adapters.evaluation.overlap_evaluator import OverlapRetrievalEvaluator
from jera.config.registry import build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import FusionMethod, Query, RetrievalMode, RetrievalResult
from jera.pipeline.corrective import CorrectiveQueryPipeline, CorrectiveResult
from jera.ports.retrieval_evaluator import RetrievalGrade

# ---------------------------------------------------------------------------
# Corpus & query constants
# ---------------------------------------------------------------------------

CORPUS = {
    "A": "cells how do organisms grow divide and multiply population",
    "B": "mitosis cytokinesis chromosomes replicate nuclear division",
}

QUERY = "how do cells divide"

# The corrective transformer bridges A's lay vocabulary to B's scientific vocabulary.
_CORRECTIVE_VARIANT = "mitosis cytokinesis nuclear division chromosomes"

# Evaluator that grades A's Jaccard (~0.44) as AMBIGUOUS.
_EVALUATOR = OverlapRetrievalEvaluator(score_threshold=0.0, overlap_threshold=0.5)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeCorrectiveTransformer:
    """Deterministic transformer: original first, then the domain-vocabulary corrective query."""

    strategy = "fake-corrective"
    version = "1.0"

    def transform(self, query: str) -> list[str]:
        return [query, _CORRECTIVE_VARIANT]


def _build_system():  # type: ignore[return]  # RagSystem has no annotation for the fixture
    system = build_system(Settings(profile=Profile.TEST))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
            for sid, md in CORPUS.items()
        ]
    )
    return system


def _source_at(result: RetrievalResult, rank: int) -> str | None:
    """Return the source_id of the chunk at position ``rank`` (0-based)."""
    if rank >= len(result.results):
        return None
    sc = result.results[rank]
    return sc.chunk.source_id if sc.chunk is not None else None


# ---------------------------------------------------------------------------
# OverlapRetrievalEvaluator — three-grade contract
# ---------------------------------------------------------------------------


def test_evaluator_grades_incorrect_when_no_results() -> None:
    """Empty retrieval → INCORRECT regardless of thresholds."""
    from jera.domain.retrieval import Query

    empty_result = RetrievalResult(
        query=Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE),
        stage="sparse",
        results=[],
    )
    assert _EVALUATOR.grade(QUERY, empty_result) is RetrievalGrade.INCORRECT


def test_evaluator_grades_incorrect_when_top_score_below_threshold() -> None:
    """Top score below score_threshold → INCORRECT (even if overlap is high).

    Uses the real ingested system so Chunk objects are fully populated — we only
    override the score via ScoredChunk to exercise the score-threshold branch.
    """
    system = _build_system()
    q = Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE)
    real = system.query.retrieve(q)
    # Re-score the top result below the threshold to trigger INCORRECT.
    strict_eval = OverlapRetrievalEvaluator(score_threshold=100.0, overlap_threshold=0.1)
    low_scored = real.model_copy(
        update={"results": [real.results[0].model_copy(update={"score": 0.5})]}
    )
    assert strict_eval.grade(QUERY, low_scored) is RetrievalGrade.INCORRECT


def test_evaluator_grades_correct_when_jaccard_meets_threshold() -> None:
    """High token overlap → CORRECT.

    Chunk A has Jaccard ~0.44 with the query.  A threshold of 0.1 lets it through.
    """
    system = _build_system()
    q = Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE)
    real = system.query.retrieve(q)
    # Top result is A (source "A") — Jaccard ~0.44 ≥ threshold 0.1 → CORRECT.
    eval_low = OverlapRetrievalEvaluator(score_threshold=0.0, overlap_threshold=0.1)
    assert eval_low.grade(QUERY, real) is RetrievalGrade.CORRECT


def test_evaluator_grades_ambiguous_when_jaccard_below_threshold() -> None:
    """Low Jaccard (below threshold) with non-negative score → AMBIGUOUS.

    Chunk A's Jaccard ~0.44 is below a threshold of 0.5 → AMBIGUOUS.
    """
    system = _build_system()
    q = Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE)
    real = system.query.retrieve(q)
    eval_strict = OverlapRetrievalEvaluator(score_threshold=0.0, overlap_threshold=0.5)
    assert eval_strict.grade(QUERY, real) is RetrievalGrade.AMBIGUOUS


# ---------------------------------------------------------------------------
# Non-tautological recall lift: B goes from NOT recall@1 → recall@1
# ---------------------------------------------------------------------------


def test_initial_retrieval_places_a_first_not_b() -> None:
    """Without correction, the literal query surfaces A (lay vocab match), not B.

    This confirms the vocabulary gap exists: B has zero BM25 score on the original
    query, so A wins the initial sparse ranking.  The corrective loop must be
    triggered to surface B.
    """
    system = _build_system()
    q = Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE)
    result = system.query.retrieve(q)

    assert _source_at(result, rank=0) == "A", "A should win the initial ranking (BM25 overlap)"
    # B has no lexical overlap → score 0.0 → not at rank 0.
    assert _source_at(result, rank=0) != "B"


def test_initial_retrieval_grades_ambiguous_not_correct() -> None:
    """The initial retrieval grades AMBIGUOUS — triggering the corrective loop.

    A's Jaccard with the query is ~0.44 (below overlap_threshold=0.5), so the
    evaluator returns AMBIGUOUS even though A is the highest-scoring result.
    """
    system = _build_system()
    q = Query(text=QUERY, top_k=2, mode=RetrievalMode.SPARSE)
    result = system.query.retrieve(q)

    grade = _EVALUATOR.grade(QUERY, result)
    assert grade in {RetrievalGrade.AMBIGUOUS, RetrievalGrade.INCORRECT}


def test_corrective_pipeline_lifts_b_to_recall_at_1() -> None:
    """Core lift test: B moves from NOT recall@1 → recall@1 after CRAG correction.

    The corrective transformer maps the lay query to scientific vocabulary
    ("mitosis cytokinesis…") that directly overlaps with chunk B.  RRF fuses the
    original and corrective rankings so B — which scored 0 on the original but #1
    on the corrective — wins the merged ranking via a tie-break (chunk_id lex asc).

    This is a genuine 0→1 lift: it requires the corrective vocabulary to bridge the
    gap.  A plain re-run of the original query (without the corrective variant) cannot
    produce this result.
    """
    system = _build_system()
    evaluator = _EVALUATOR
    transformer = _FakeCorrectiveTransformer()
    pipe = CorrectiveQueryPipeline(system.query, evaluator, transformer)

    result: CorrectiveResult = pipe.answer(
        QUERY, top_k=2, mode=RetrievalMode.SPARSE, fusion=FusionMethod.RRF
    )

    assert result.corrected is True, "Corrective loop should have fired (grade != CORRECT)"
    assert result.grade in {RetrievalGrade.AMBIGUOUS, RetrievalGrade.INCORRECT}
    assert result.contexts, "Corrected retrieval must return at least one context"
    assert result.contexts[0].source_id == "B", (
        f"B should be recall@1 after correction; got {result.contexts[0].source_id!r}.  "
        "The corrective vocabulary must bridge the lay-query ↔ domain-term gap."
    )


def test_corrective_pipeline_does_not_correct_when_grade_is_correct() -> None:
    """When the initial retrieval is CORRECT the pipeline skips the corrective loop."""
    system = _build_system()
    # Low threshold: any non-empty result with non-negative score grades CORRECT.
    permissive_eval = OverlapRetrievalEvaluator(score_threshold=0.0, overlap_threshold=0.0)
    transformer = _FakeCorrectiveTransformer()
    pipe = CorrectiveQueryPipeline(system.query, permissive_eval, transformer)

    result: CorrectiveResult = pipe.answer(
        QUERY, top_k=2, mode=RetrievalMode.SPARSE, fusion=FusionMethod.RRF
    )

    assert result.corrected is False, "No correction should happen when grade is CORRECT"
    assert result.grade is RetrievalGrade.CORRECT


def test_corrective_result_exposes_answer_and_grade() -> None:
    """CorrectiveResult has .answer (Answer), .grade (RetrievalGrade), .corrected (bool)."""
    system = _build_system()
    pipe = CorrectiveQueryPipeline(system.query, _EVALUATOR, _FakeCorrectiveTransformer())
    result: CorrectiveResult = pipe.answer(
        QUERY, top_k=2, mode=RetrievalMode.SPARSE, fusion=FusionMethod.RRF
    )

    assert hasattr(result.answer, "text")
    assert isinstance(result.grade, RetrievalGrade)
    assert isinstance(result.corrected, bool)
