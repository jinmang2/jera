"""Pricing table + snapshot cost-metadata (replaces the old placeholder)."""

from __future__ import annotations

from jera.config.pricing import (
    pricing_for,
    snapshot_cost_metadata,
)
from jera.config.registry import build_system
from jera.config.settings import Profile, Settings


def test_local_deterministic_models_are_free() -> None:
    for mid in ("hash-emb-v1", "bm25-local-v1", "extractive-v1", "identity-rerank-v1"):
        p = pricing_for(mid)
        assert p is not None and p.free_local


def test_cloud_models_carry_real_prices() -> None:
    opus = pricing_for("claude-opus-4-8")
    assert opus is not None and opus.input_per_mtok > 0 and opus.output_per_mtok > 0
    emb = pricing_for("text-embedding-3-small")
    assert emb is not None and emb.input_per_mtok > 0
    rerank = pricing_for("rerank-v3.5")
    assert rerank is not None and rerank.per_1k_searches > 0


def test_fastembed_local_repo_ids_are_free() -> None:
    # local ONNX model ids look like "BAAI/bge-m3" — run locally, no per-call cost
    p = pricing_for("BAAI/bge-m3")
    assert p is not None and p.free_local


def test_unknown_cloud_id_is_unpriced_not_crashing() -> None:
    assert pricing_for("some-unreleased-model") is None


def test_snapshot_cost_metadata_shape() -> None:
    meta = snapshot_cost_metadata(
        embedding="text-embedding-3-small",
        sparse="bm25-local-v1",
        reranker="rerank-v3.5",
        generator="claude-opus-4-8",
    )
    assert meta["currency"] == "USD"
    assert "as_of" in meta
    for slot in ("embedding", "sparse", "reranker", "generator"):
        assert "model_id" in meta[slot]  # type: ignore[operator]


def test_test_profile_snapshot_records_free_costs_not_a_placeholder() -> None:
    system = build_system(Settings(profile=Profile.TEST))
    snap = system.metadata_store.latest_config_snapshot()
    assert snap is not None
    cost = snap.cost_metadata
    # The old placeholder string must be gone; real structured pricing must be present.
    assert "note" not in cost or "placeholder" not in str(cost.get("note", ""))
    assert cost["currency"] == "USD"
    embedding = cost["embedding"]
    assert isinstance(embedding, dict) and embedding["pricing"] == {"free_local": True}
