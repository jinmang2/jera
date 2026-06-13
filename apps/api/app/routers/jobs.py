"""Jobs router — exposes ingestion-job status for polling."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_system
from app.schemas import JobResponse
from jera.rag import RagSystem

router = APIRouter(tags=["jobs"])

SystemDep = Annotated[RagSystem, Depends(get_system)]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, system: SystemDep) -> JobResponse:
    job = system.metadata_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return JobResponse(
        job_id=job.job_id,
        source_id=job.source_id,
        status=job.status.value,
        document_id=job.document_id,
        chunk_count=job.chunk_count,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
