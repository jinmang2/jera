"""Multi-query retrieval (QueryPipeline.retrieve_multi) — RRF over query variants.

Headline (non-tautological): a real answer that the *literal* query cannot retrieve (zero
lexical score) is surfaced as the top result once a HyDE hypothesis bridges the vocabulary gap.
Plus the graceful-fallback contracts (no transformer / single variant == plain retrieve).
"""

from __future__ import annotations

from jera.adapters.query.hyde import HydeTransformer
from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import Query, RetrievalMode
from jera.pipeline.query import QueryPipeline

# The real answer is written in domain vocabulary the user's phrasing never mentions; `lit` is an
# FAQ that echoes the literal question but answers nothing.
CORPUS = {
    "answer": "# Store\n\nPersistence uses an append-only log compacted into immutable sstables.\n",
    "lit": "# FAQ\n\nWhere is my data kept? This page asks where data is kept, no detail.\n",
    "d2": "# d2\n\nThe dashboard renders charts from metrics.\n",
}
QUERY = "where is my data kept"


class _FakeHyDE:
    """Deterministic HyDE: a hypothesis written in the answer's domain vocabulary (no paid call)."""

    model_id = "fake-hyde"

    def hypothesize(self, query: str) -> str:
        return "Persistence relies on an append-only log compacted into immutable sstables."


def _ingested_system() -> RagSystem:
    system = build_system(Settings(profile=Profile.TEST))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
            for sid, md in CORPUS.items()
        ]
    )
    return system


def _pipeline(system: RagSystem, *, transform: bool) -> QueryPipeline:
    return QueryPipeline(
        embedding=system.embedding,
        sparse=system.sparse,
        vector_store=system.vector_store,
        metadata_store=system.metadata_store,
        reranker=system.reranker,
        generator=system.generator,
        collection="jera_chunks",
        query_transformer=HydeTransformer(_FakeHyDE()) if transform else None,
    )


def _by_source(system: RagSystem, *, transform: bool) -> list[tuple[str, float]]:
    result = _pipeline(system, transform=transform).retrieve_multi(
        Query(text=QUERY, top_k=3, mode=RetrievalMode.SPARSE)
    )
    return [(sc.chunk.source_id, sc.score) for sc in result.results if sc.chunk]


def test_literal_query_cannot_retrieve_the_real_answer() -> None:
    ranked = _by_source(_ingested_system(), transform=False)
    by_source = dict(ranked)
    # The literal-phrasing FAQ wins; the real answer has zero lexical overlap → score 0.0.
    assert ranked[0][0] == "lit"
    assert by_source["answer"] == 0.0


def test_hyde_surfaces_the_answer_the_literal_query_missed() -> None:
    ranked = _by_source(_ingested_system(), transform=True)
    by_source = dict(ranked)
    # The hypothesis bridges the vocabulary gap: the answer is now genuinely scored (>0) AND first,
    # outranking the literal FAQ by a real margin (not just a tie-break).
    assert ranked[0][0] == "answer"
    assert by_source["answer"] > 0.0
    assert by_source["answer"] > by_source["lit"]


def test_no_transformer_falls_back_to_single_query() -> None:
    system = _ingested_system()
    q = Query(text=QUERY, top_k=3, mode=RetrievalMode.SPARSE)
    plain = _pipeline(system, transform=False)
    multi = plain.retrieve_multi(q)
    single = plain.retrieve(q)
    assert multi.stage == "sparse"  # no transformer → identical to retrieve()
    assert [c.chunk_id for c in multi.results] == [c.chunk_id for c in single.results]


def test_single_variant_transformer_falls_back() -> None:
    class NoOp:
        strategy = "noop"
        version = "1.0"

        def transform(self, query: str) -> list[str]:
            return [query]

    system = _ingested_system()
    pipe = QueryPipeline(
        embedding=system.embedding,
        sparse=system.sparse,
        vector_store=system.vector_store,
        metadata_store=system.metadata_store,
        reranker=system.reranker,
        generator=system.generator,
        collection="jera_chunks",
        query_transformer=NoOp(),
    )
    result = pipe.retrieve_multi(Query(text=QUERY, top_k=3, mode=RetrievalMode.SPARSE))
    assert result.stage == "sparse"  # only the original variant → single-query path


def test_mmr_reranker_profile_answers_end_to_end() -> None:
    # The MMR reranker (wired via reranker_kind) works through the full answer path.
    system = build_system(Settings(profile=Profile.TEST, reranker_kind="mmr"))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
            for sid, md in CORPUS.items()
        ]
    )
    answer = system.query.answer("append-only log sstables", top_k=3, mode=RetrievalMode.SPARSE)
    assert answer.citations  # produced a grounded answer without error
