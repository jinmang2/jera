"""Evaluation dataset contracts (datasets themselves are a later, criterion-gated milestone)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CaseKind(StrEnum):
    RETRIEVAL = "retrieval"
    TABLE = "table"
    COMPUTATION = "computation"


class GoldChunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    relevance: float = 1.0


class EvalCase(BaseModel):
    model_config = {"frozen": True}

    case_id: str
    query: str
    gold: list[GoldChunk]

    # case type — default keeps existing retrieval cases backward-compatible
    kind: CaseKind = CaseKind.RETRIEVAL

    # computation fields (ComputationQ only)
    expected_value: float | None = None
    tolerance: float = 0.001
    formula: str | None = None
    cited_numbers: list[float] = Field(default_factory=list)

    # attribution — carried from the corpus manifest into committed JSON
    source_inst: str | None = None
    source_url: str | None = None
    license: str | None = None


class EvalDataset(BaseModel):
    name: str
    cases: list[EvalCase]
