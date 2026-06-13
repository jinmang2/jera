"""Per-model pricing — replaces the placeholder ``cost_metadata`` with real list prices.

Indicative published list prices (USD), captured ``AS_OF`` below. Deterministic local/offline
models (hash embedding, BM25, in-memory store, extractive/identity/MMR) are free. Cloud models
carry their token/search prices. The ``ProviderConfigSnapshot.cost_metadata`` is populated from
here so a profile's running cost is recorded with its config, not left as a TODO.

Prices change; treat this as a dated reference, not a billing oracle. Update ``AS_OF`` + the
table when revising. (Sources: Anthropic / OpenAI / Cohere public pricing pages.)
"""

from __future__ import annotations

from dataclasses import dataclass

AS_OF = "2026-06"
CURRENCY = "USD"


@dataclass(frozen=True)
class ModelPricing:
    """List price for one model. Token prices are per-million-tokens (MTok)."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    per_1k_searches: float = 0.0
    free_local: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        if self.free_local:
            return {"free_local": True}
        out: dict[str, float | bool] = {}
        if self.input_per_mtok:
            out["input_per_mtok"] = self.input_per_mtok
        if self.output_per_mtok:
            out["output_per_mtok"] = self.output_per_mtok
        if self.per_1k_searches:
            out["per_1k_searches"] = self.per_1k_searches
        return out or {"free_local": True}


_FREE = ModelPricing(free_local=True)

# Keyed by the exact ``model_id`` each adapter reports.
PRICES: dict[str, ModelPricing] = {
    # --- Anthropic (input / output per MTok) ---
    # Source: https://platform.claude.com/docs/en/docs/about-claude/models (verified 2026-06)
    # claude-opus-4-8: $5/$25 — NOT $15/$75 (those were deprecated claude-opus-4-1/4-0)
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-sonnet-4-6": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5-20251001": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0),
    # --- OpenAI embeddings (input per MTok) ---
    # Source: developers.openai.com/api/docs/models/text-embedding-3-{small,large} (2026-06)
    "text-embedding-3-small": ModelPricing(input_per_mtok=0.02),
    "text-embedding-3-large": ModelPricing(input_per_mtok=0.13),
    # --- Cohere rerank (per 1K searches) ---
    # Source: cohere.com/pricing — per-search rate not publicly listed at 2026-06;
    # $2.00/1k is a previously published indicative rate (treat as approximate).
    "rerank-v3.5": ModelPricing(per_1k_searches=2.0),
}

# Deterministic/local model_ids are free. Many embed a suffix (e.g. "hash-emb-v1-256") or are
# fakes ("fake-tool-use-v1"), so match by prefix rather than exact id.
_FREE_PREFIXES = (
    "hash-emb",
    "bm25-local",
    "extractive",
    "identity-rerank",
    "mmr-rerank",
    "fake-",
)


def pricing_for(model_id: str) -> ModelPricing | None:
    """Look up a model's pricing. Local/deterministic and fastembed/ONNX model ids are free;
    unknown cloud ids return None (recorded as 'unpriced')."""
    if model_id in PRICES:
        return PRICES[model_id]
    # fastembed/local model ids look like "BAAI/bge-m3" etc. — run locally, no per-call cost.
    if "/" in model_id or model_id.startswith(_FREE_PREFIXES):
        return _FREE
    return None


def _entry(model_id: str) -> dict[str, object]:
    p = pricing_for(model_id)
    if p is None:
        return {"model_id": model_id, "pricing": "unpriced"}
    return {"model_id": model_id, "pricing": p.as_dict()}


def snapshot_cost_metadata(
    *, embedding: str, sparse: str, reranker: str, generator: str
) -> dict[str, object]:
    """Build the ``ProviderConfigSnapshot.cost_metadata`` for a wired profile."""
    return {
        "as_of": AS_OF,
        "currency": CURRENCY,
        "embedding": _entry(embedding),
        "sparse": _entry(sparse),
        "reranker": _entry(reranker),
        "generator": _entry(generator),
    }
