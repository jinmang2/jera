"""QueryStats: per-stage timing + cost estimation on the query pipeline.

Tests:
- stats populated on a real (test-profile) round-trip
- timings_ms has the 4 expected keys, all >= 0, total >= each stage
- model_ids populated
- estimated_cost_usd == 0.0 for the deterministic free-local test profile
- empty-retrieval path also produces valid stats (zero cost)
- focused unit test of estimate_query_cost with a priced model id (claude-opus-4-8)
"""

from __future__ import annotations

import pytest

from jera.config.pricing import ModelPricing, pricing_for
from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.pipeline.query import QueryStats, estimate_query_cost

SAMPLE_MARKDOWN = """# Retrieval Overview

Jera supports dense, sparse, and hybrid retrieval.
Hybrid mode uses reciprocal rank fusion to merge rankings.
The ranking module identifier is ZX9000.
"""

_TIMING_KEYS = {"retrieve", "rerank", "generate", "total"}


@pytest.fixture
def system() -> RagSystem:
    return build_system(Settings(profile=Profile.TEST))


@pytest.fixture
def ingested_system(system: RagSystem) -> RagSystem:
    system.ingest.ingest(
        SourceRef(
            source_id="stats-doc",
            media_type=MediaType.MARKDOWN,
            content=SAMPLE_MARKDOWN.encode(),
        )
    )
    return system


# ---------------------------------------------------------------------------
# Full round-trip stats
# ---------------------------------------------------------------------------


def test_stats_populated_after_answer_with_contexts(ingested_system: RagSystem) -> None:
    result = ingested_system.query.answer_with_contexts("What is hybrid retrieval?", top_k=3)
    assert result.stats is not None


def test_stats_timings_keys_present(ingested_system: RagSystem) -> None:
    result = ingested_system.query.answer_with_contexts("What is hybrid retrieval?", top_k=3)
    assert result.stats is not None
    assert set(result.stats.timings_ms.keys()) == _TIMING_KEYS


def test_stats_timings_all_non_negative(ingested_system: RagSystem) -> None:
    result = ingested_system.query.answer_with_contexts("hybrid retrieval ranking", top_k=3)
    assert result.stats is not None
    for key, val in result.stats.timings_ms.items():
        assert val >= 0.0, f"timing '{key}' is negative: {val}"


def test_stats_total_ge_each_stage(ingested_system: RagSystem) -> None:
    result = ingested_system.query.answer_with_contexts("ZX9000 ranking module", top_k=3)
    assert result.stats is not None
    total = result.stats.timings_ms["total"]
    for key in ("retrieve", "rerank", "generate"):
        assert total >= result.stats.timings_ms[key], (
            f"total {total:.3f}ms < stage '{key}' {result.stats.timings_ms[key]:.3f}ms"
        )


def test_stats_model_ids_populated(ingested_system: RagSystem) -> None:
    result = ingested_system.query.answer_with_contexts("dense sparse hybrid", top_k=3)
    assert result.stats is not None
    assert "embedding" in result.stats.model_ids
    assert "reranker" in result.stats.model_ids
    assert "generator" in result.stats.model_ids
    for key, val in result.stats.model_ids.items():
        assert isinstance(val, str) and val, f"model_ids['{key}'] is empty"


def test_stats_cost_zero_for_free_local_test_profile(ingested_system: RagSystem) -> None:
    """The TEST profile uses hash-embedding + extractive generator — both free_local."""
    result = ingested_system.query.answer_with_contexts("reciprocal rank fusion", top_k=3)
    assert result.stats is not None
    assert result.stats.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Empty-retrieval path
# ---------------------------------------------------------------------------


def test_stats_populated_on_empty_retrieval(system: RagSystem) -> None:
    """Empty index → empty answer; stats must still be populated with zero cost."""
    result = system.query.answer_with_contexts("anything at all", top_k=3)
    assert result.stats is not None
    assert set(result.stats.timings_ms.keys()) == _TIMING_KEYS
    assert result.stats.estimated_cost_usd == 0.0
    assert result.stats.timings_ms["retrieve"] >= 0.0
    assert result.stats.timings_ms["total"] >= result.stats.timings_ms["retrieve"]


# ---------------------------------------------------------------------------
# answer() return type unchanged
# ---------------------------------------------------------------------------


def test_answer_still_returns_answer_object(ingested_system: RagSystem) -> None:
    from jera.domain.answer import Answer

    ans = ingested_system.query.answer("hybrid retrieval", top_k=3)
    assert isinstance(ans, Answer)


# ---------------------------------------------------------------------------
# Pure unit test of estimate_query_cost — priced model
# ---------------------------------------------------------------------------


def test_estimate_query_cost_zero_for_free_local() -> None:
    free = ModelPricing(free_local=True)
    cost = estimate_query_cost(
        query_text="hello world",
        context_texts=["some context text"],
        answer_text="the answer",
        embedding_pricing=free,
        generator_pricing=free,
    )
    assert cost == 0.0


def test_estimate_query_cost_zero_when_pricing_none() -> None:
    cost = estimate_query_cost(
        query_text="hello world",
        context_texts=["some context text"],
        answer_text="the answer",
        embedding_pricing=None,
        generator_pricing=None,
    )
    assert cost == 0.0


def test_estimate_query_cost_positive_for_priced_generator() -> None:
    """claude-opus-4-8 carries real token prices → cost must be > 0."""
    gen_pricing = pricing_for("claude-opus-4-8")
    assert gen_pricing is not None
    cost = estimate_query_cost(
        query_text="What is retrieval augmented generation?",
        context_texts=["RAG combines retrieval with generation to produce grounded answers."],
        answer_text="Retrieval augmented generation grounds LLM answers in retrieved documents.",
        embedding_pricing=None,
        generator_pricing=gen_pricing,
    )
    assert cost > 0.0


def test_estimate_query_cost_math_correctness() -> None:
    """Verify the arithmetic directly against the formula: chars/4/1e6 * price."""
    gen_pricing = ModelPricing(input_per_mtok=15.0, output_per_mtok=75.0)
    query = "hello"  # 5 chars
    context = "world"  # 5 chars
    answer = "ok"  # 2 chars

    # prompt_tokens = (5 + 5) / 4 = 2.5; completion_tokens = 2 / 4 = 0.5
    expected = (10 / 4.0) / 1e6 * 15.0 + (2 / 4.0) / 1e6 * 75.0
    cost = estimate_query_cost(
        query_text=query,
        context_texts=[context],
        answer_text=answer,
        embedding_pricing=None,
        generator_pricing=gen_pricing,
    )
    assert abs(cost - expected) < 1e-12


def test_estimate_query_cost_embedding_contributes() -> None:
    """Embedding pricing adds query-token cost on top of generator cost."""
    emb_pricing = ModelPricing(input_per_mtok=0.02)
    gen_pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
    query = "hello"

    cost_with_emb = estimate_query_cost(
        query_text=query,
        context_texts=[],
        answer_text="",
        embedding_pricing=emb_pricing,
        generator_pricing=gen_pricing,
    )
    cost_without_emb = estimate_query_cost(
        query_text=query,
        context_texts=[],
        answer_text="",
        embedding_pricing=None,
        generator_pricing=gen_pricing,
    )
    assert cost_with_emb > cost_without_emb


def test_query_stats_is_frozen_pydantic() -> None:
    """QueryStats must be an immutable (frozen) pydantic model."""
    stats = QueryStats(
        timings_ms={"retrieve": 1.0, "rerank": 2.0, "generate": 3.0, "total": 6.0},
        estimated_cost_usd=0.0,
        model_ids={
            "embedding": "hash-emb-v1-256",
            "reranker": "identity-rerank-v1",
            "generator": "extractive-v1",
        },
    )
    with pytest.raises((TypeError, Exception)):
        stats.estimated_cost_usd = 99.0  # type: ignore[misc]
