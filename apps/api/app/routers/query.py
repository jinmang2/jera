"""Query router — maps API requests onto jera.rag's QueryPipeline."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_system
from app.schemas import CitationOut, QueryRequest, QueryResponse, QueryStatsOut
from jera.rag import RagSystem

router = APIRouter(tags=["query"])

SystemDep = Annotated[RagSystem, Depends(get_system)]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, system: SystemDep) -> QueryResponse:
    result = system.query.answer_with_contexts(
        req.query, top_k=req.top_k, mode=req.mode, fusion=req.fusion
    )
    answer = result.answer
    stats = (
        QueryStatsOut(
            timings_ms=result.stats.timings_ms,
            estimated_cost_usd=result.stats.estimated_cost_usd,
            model_ids=result.stats.model_ids,
        )
        if result.stats is not None
        else None
    )
    return QueryResponse(
        query=answer.query,
        text=answer.text,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                snippet=c.snippet,
                score=c.score,
                page_span=c.page_span,
                section_path=c.section_path,
            )
            for c in answer.citations
        ],
        stats=stats,
    )
