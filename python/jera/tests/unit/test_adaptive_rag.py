"""Adaptive-RAG routing — HeuristicQueryRouter + AdaptiveQueryPipeline.

Non-tautological headline: a counting proxy wraps the VectorStore and asserts that
NO_RETRIEVAL genuinely triggers ZERO ``search`` calls (real token/compute saving),
while SINGLE_STEP triggers ≥ 1 call, and MULTI_STEP (with a transformer set) triggers
strictly MORE calls than a bare SINGLE_STEP query.

The proxy counting is not mocked away — if the pipeline calls ``search``, the counter
increments regardless of what the router decided, so a bug in the NO_RETRIEVAL path
that accidentally calls ``search`` will make the test fail.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.adapters.query.heuristic_router import HeuristicQueryRouter
from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import FusionMethod, RetrievalMode, ScoredChunk
from jera.domain.vectors import DenseVector, SparseVector
from jera.pipeline.adaptive import AdaptiveAnsweredQuery, AdaptiveQueryPipeline
from jera.pipeline.query import QueryPipeline
from jera.ports.query_router import QueryComplexity
from jera.ports.query_transformer import QueryTransformer
from jera.ports.vector_store import CollectionSpec, VectorRecord, VectorStore

# ---------------------------------------------------------------------------
# Counting proxy
# ---------------------------------------------------------------------------


class _CountingVectorStore:
    """Transparent proxy around a real VectorStore that counts ``search`` calls."""

    def __init__(self, real: VectorStore) -> None:
        self._real = real
        self.search_call_count: int = 0

    # -- VectorStore protocol --

    def ensure_collection(self, spec: CollectionSpec) -> None:
        self._real.ensure_collection(spec)

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        self._real.upsert(collection, records)

    def delete(self, collection: str, chunk_ids: Sequence[str]) -> None:
        self._real.delete(collection, chunk_ids)

    def search(
        self,
        collection: str,
        *,
        dense: DenseVector | None = None,
        sparse: SparseVector | None = None,
        top_k: int = 10,
        fusion: FusionMethod = FusionMethod.RRF,
        prefetch_limit: int = 100,
    ) -> list[ScoredChunk]:
        self.search_call_count += 1
        return self._real.search(
            collection,
            dense=dense,
            sparse=sparse,
            top_k=top_k,
            fusion=fusion,
            prefetch_limit=prefetch_limit,
        )


# ---------------------------------------------------------------------------
# Minimal corpus + test-profile system builder
# ---------------------------------------------------------------------------

_CORPUS = {
    "a": "# Dense retrieval\n\nDense retrieval uses embedding vectors for semantic search.\n",
    "b": "# Sparse retrieval\n\nSparse retrieval uses BM25 for keyword matching.\n",
    "c": "# Hybrid retrieval\n\nHybrid fuses dense and sparse rankings via RRF.\n",
}


def _build_ingested_system() -> RagSystem:
    system = build_system(Settings(profile=Profile.TEST))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
            for sid, md in _CORPUS.items()
        ]
    )
    return system


def _pipeline_with_counter(
    system: RagSystem,
    *,
    counting_store: _CountingVectorStore,
    query_transformer: QueryTransformer | None = None,
) -> QueryPipeline:
    """Build a QueryPipeline that uses *counting_store* instead of the system's real store."""
    return QueryPipeline(
        embedding=system.embedding,
        sparse=system.sparse,
        vector_store=counting_store,
        metadata_store=system.metadata_store,
        reranker=system.reranker,
        generator=system.generator,
        collection=system.collection,
        query_transformer=query_transformer,
    )


# ---------------------------------------------------------------------------
# Part (a): Router verdict unit tests — three representative queries
# ---------------------------------------------------------------------------


def test_router_no_retrieval_short_query() -> None:
    router = HeuristicQueryRouter()
    # 3 tokens → ≤ 4 token threshold → NO_RETRIEVAL
    assert router.route("what is RAG") == QueryComplexity.NO_RETRIEVAL


def test_router_no_retrieval_arithmetic() -> None:
    router = HeuristicQueryRouter()
    # Pure arithmetic pattern → NO_RETRIEVAL regardless of length
    assert router.route("2 + 2") == QueryComplexity.NO_RETRIEVAL


def test_router_single_step_factual_query() -> None:
    router = HeuristicQueryRouter()
    # Factual question, no multi-hop cues, more than 4 tokens → SINGLE_STEP
    verdict = router.route("what is dense retrieval in a RAG system")
    assert verdict == QueryComplexity.SINGLE_STEP


def test_router_single_step_descriptive_query() -> None:
    router = HeuristicQueryRouter()
    verdict = router.route("explain sparse retrieval using BM25 scoring")
    assert verdict == QueryComplexity.SINGLE_STEP


def test_router_multi_step_compare_cue() -> None:
    router = HeuristicQueryRouter()
    verdict = router.route("compare dense retrieval versus sparse retrieval")
    assert verdict == QueryComplexity.MULTI_STEP


def test_router_multi_step_difference_between_cue() -> None:
    router = HeuristicQueryRouter()
    verdict = router.route("what is the difference between BM25 and dense embeddings")
    assert verdict == QueryComplexity.MULTI_STEP


def test_router_multi_step_relationship_cue() -> None:
    router = HeuristicQueryRouter()
    verdict = router.route("describe the relationship between precision and recall in retrieval")
    assert verdict == QueryComplexity.MULTI_STEP


def test_router_multi_step_case_insensitive() -> None:
    router = HeuristicQueryRouter()
    # Cue matching must be case-insensitive
    verdict = router.route("COMPARE dense and sparse retrieval methods")
    assert verdict == QueryComplexity.MULTI_STEP


def test_router_custom_cues() -> None:
    router = HeuristicQueryRouter(
        multi_hop_cues=frozenset({"synergy"}),
        no_retrieval_max_tokens=2,
    )
    assert router.route("synergy between modules") == QueryComplexity.MULTI_STEP
    assert router.route("ok") == QueryComplexity.NO_RETRIEVAL
    # 5 tokens, no custom cue → SINGLE_STEP
    assert router.route("how does dense retrieval work") == QueryComplexity.SINGLE_STEP


# ---------------------------------------------------------------------------
# Part (b): Non-tautological search-call counting
# ---------------------------------------------------------------------------


def test_no_retrieval_triggers_zero_search_calls() -> None:
    """NO_RETRIEVAL must bypass the vector store entirely — zero ``search`` calls.

    This is the core Adaptive-RAG saving: for self-contained queries the embedding
    call, BM25 scoring, and vector-store round-trip are all skipped.  The counting
    proxy ensures this is real, not just a routing label.
    """
    system = _build_ingested_system()
    counting_store = _CountingVectorStore(system.vector_store)
    pipe = _pipeline_with_counter(system, counting_store=counting_store)
    router = HeuristicQueryRouter()
    adaptive = AdaptiveQueryPipeline(pipe, router)

    # "hi" → 1 token → NO_RETRIEVAL
    result = adaptive.answer("hi", top_k=3, mode=RetrievalMode.SPARSE)

    assert result.complexity == QueryComplexity.NO_RETRIEVAL
    assert counting_store.search_call_count == 0, (
        f"Expected 0 search calls for NO_RETRIEVAL but got {counting_store.search_call_count}"
    )


def test_single_step_triggers_at_least_one_search_call() -> None:
    """SINGLE_STEP must actually query the vector store (≥ 1 ``search`` call)."""
    system = _build_ingested_system()
    counting_store = _CountingVectorStore(system.vector_store)
    pipe = _pipeline_with_counter(system, counting_store=counting_store)
    router = HeuristicQueryRouter()
    adaptive = AdaptiveQueryPipeline(pipe, router)

    result = adaptive.answer(
        "explain how dense retrieval uses embedding vectors for semantic search",
        top_k=3,
        mode=RetrievalMode.SPARSE,
    )

    assert result.complexity == QueryComplexity.SINGLE_STEP
    assert counting_store.search_call_count >= 1, (
        f"Expected ≥ 1 search call for SINGLE_STEP but got {counting_store.search_call_count}"
    )


def test_multi_step_with_transformer_triggers_more_searches_than_single_step() -> None:
    """MULTI_STEP (with transformer) must trigger strictly more ``search`` calls
    than SINGLE_STEP.

    The ``RuleBasedExpander`` splits the compound query into clauses and retrieves
    each independently, so the count rises above the single-variant baseline.
    This is the multi-hop benefit: more diverse evidence is gathered.
    """
    from jera.adapters.query.rule_based_expander import RuleBasedExpander

    system = _build_ingested_system()
    transformer = RuleBasedExpander()

    # -- Baseline: SINGLE_STEP search count (no transformer) --
    single_store = _CountingVectorStore(system.vector_store)
    single_pipe = _pipeline_with_counter(system, counting_store=single_store)
    single_router = HeuristicQueryRouter()
    single_adaptive = AdaptiveQueryPipeline(single_pipe, single_router)
    single_result = single_adaptive.answer(
        "explain how dense retrieval works in a RAG system",
        top_k=2,
        mode=RetrievalMode.SPARSE,
    )
    assert single_result.complexity == QueryComplexity.SINGLE_STEP
    single_count = single_store.search_call_count

    # -- MULTI_STEP: with transformer (compound query splits into multiple clauses) --
    multi_store = _CountingVectorStore(system.vector_store)
    multi_pipe = _pipeline_with_counter(
        system, counting_store=multi_store, query_transformer=transformer
    )
    multi_router = HeuristicQueryRouter()
    multi_adaptive = AdaptiveQueryPipeline(multi_pipe, multi_router)
    # "compare … and …" → MULTI_STEP; RuleBasedExpander will split at "and" into 2+ clauses
    multi_result = multi_adaptive.answer(
        "compare dense retrieval and sparse retrieval methods",
        top_k=2,
        mode=RetrievalMode.SPARSE,
    )
    assert multi_result.complexity == QueryComplexity.MULTI_STEP
    multi_count = multi_store.search_call_count

    assert multi_count > single_count, (
        f"MULTI_STEP with transformer ({multi_count} searches) should exceed "
        f"SINGLE_STEP ({single_count} searches)"
    )


def test_no_retrieval_answer_is_returned_without_contexts() -> None:
    """NO_RETRIEVAL path returns an AnsweredQuery with empty contexts and retrieved_ids."""
    system = _build_ingested_system()
    counting_store = _CountingVectorStore(system.vector_store)
    pipe = _pipeline_with_counter(system, counting_store=counting_store)
    adaptive = AdaptiveQueryPipeline(pipe, HeuristicQueryRouter())

    result: AdaptiveAnsweredQuery = adaptive.answer("ok", top_k=3, mode=RetrievalMode.SPARSE)

    assert result.complexity == QueryComplexity.NO_RETRIEVAL
    assert result.answered_query.contexts == []
    assert result.answered_query.retrieved_ids == []
    assert counting_store.search_call_count == 0


def test_adaptive_answered_query_exposes_answer_and_complexity() -> None:
    """AdaptiveAnsweredQuery must expose both .answer and .complexity on all paths."""
    system = _build_ingested_system()
    pipe = _pipeline_with_counter(system, counting_store=_CountingVectorStore(system.vector_store))
    adaptive = AdaptiveQueryPipeline(pipe, HeuristicQueryRouter())

    result = adaptive.answer(
        "explain how dense retrieval works in a RAG system",
        top_k=2,
        mode=RetrievalMode.SPARSE,
    )
    # .answer is an Answer domain object
    assert hasattr(result.answer, "text")
    assert hasattr(result.answer, "citations")
    # .complexity is a QueryComplexity enum member
    assert isinstance(result.complexity, QueryComplexity)
