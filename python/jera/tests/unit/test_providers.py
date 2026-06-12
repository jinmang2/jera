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
    assert "note" in snap.cost_metadata  # pricing placeholder present
