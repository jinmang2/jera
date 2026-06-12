"""Evaluation dataset contracts (datasets themselves are a later, criterion-gated milestone)."""

from __future__ import annotations

from pydantic import BaseModel


class GoldChunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    relevance: float = 1.0


class EvalCase(BaseModel):
    model_config = {"frozen": True}

    case_id: str
    query: str
    gold: list[GoldChunk]


class EvalDataset(BaseModel):
    name: str
    cases: list[EvalCase]
