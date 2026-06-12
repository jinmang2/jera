"""Ingest router — maps API requests onto jera.rag's IngestPipeline."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_system
from app.schemas import IngestRequest, IngestResponse
from jera.rag import RagSystem, SourceRef

router = APIRouter(tags=["ingest"])

SystemDep = Annotated[RagSystem, Depends(get_system)]


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, system: SystemDep) -> IngestResponse:
    if req.text is not None:
        content = req.text.encode("utf-8")
    elif req.content_b64 is not None:
        content = base64.b64decode(req.content_b64)
    else:
        raise HTTPException(status_code=422, detail="provide either `text` or `content_b64`")

    source = SourceRef(
        source_id=req.source_id,
        media_type=req.media_type,
        content=content,
        filename=req.filename,
    )
    job = system.ingest.ingest(source)
    return IngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        document_id=job.document_id,
        chunk_count=job.chunk_count,
    )
