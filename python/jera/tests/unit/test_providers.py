"""Provider gate (Gate 6): paid providers disabled by default; config snapshot records identity."""

from __future__ import annotations

import pytest

from jera.config.registry import build_system
from jera.config.settings import Profile, Settings


def test_openai_embedding_disabled_by_default() -> None:
    from jera.adapters.embedding.openai_embedding import OpenAIEmbedding

    with pytest.raises(RuntimeError, match="disabled by default"):
        OpenAIEmbedding()


def test_cohere_reranker_disabled_by_default() -> None:
    from jera.adapters.ranking.cohere_reranker import CohereReranker

    with pytest.raises(RuntimeError, match="disabled by default"):
        CohereReranker()


def test_claude_generator_disabled_by_default() -> None:
    from jera.adapters.generator.claude_generator import ClaudeGenerator

    with pytest.raises(RuntimeError, match="disabled by default"):
        ClaudeGenerator()


def test_test_profile_records_config_snapshot_with_identity() -> None:
    system = build_system(Settings(profile=Profile.TEST))
    snap = system.metadata_store.latest_config_snapshot()
    assert snap is not None
    assert snap.profile == "test"
    assert snap.embedding_model_id == "hash-emb-v1-256"
    assert snap.embedding_dimensions == 256
    assert snap.sparse_model_id == "bm25-local-v1"
    # Real structured pricing (no longer a placeholder note): local models recorded as free.
    assert snap.cost_metadata["currency"] == "USD"
    sparse_cost = snap.cost_metadata["sparse"]
    assert isinstance(sparse_cost, dict) and sparse_cost["pricing"] == {"free_local": True}


def test_generator_kind_tooluse_builds_tool_augmented_generator() -> None:
    """generator_kind='tooluse' wires a ToolAugmentedGenerator (offline, zero API)."""
    from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator

    system = build_system(Settings(profile=Profile.TEST, generator_kind="tooluse"))
    assert isinstance(system.generator, ToolAugmentedGenerator)


def test_generator_kind_extractive_is_default() -> None:
    """Default generator_kind produces the extractive generator."""
    from jera.adapters.generator.extractive_generator import ExtractiveGenerator

    system = build_system(Settings(profile=Profile.TEST))
    assert isinstance(system.generator, ExtractiveGenerator)


def test_settings_embedding_model_override_stored() -> None:
    """embedding_model field round-trips through Settings."""
    s = Settings(profile=Profile.TEST, embedding_model="BAAI/bge-m3")
    assert s.embedding_model == "BAAI/bge-m3"


def test_settings_reranker_model_override_stored() -> None:
    """reranker_model field round-trips through Settings."""
    s = Settings(profile=Profile.TEST, reranker_model="BAAI/bge-reranker-v2-m3")
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
