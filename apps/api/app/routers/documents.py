"""Documents router — list, inspect, and delete ingested documents.

Deletion is full-lifecycle: it removes the document + its chunks from the metadata store and
the matching vectors from the vector store, so a re-ingest starts clean.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_system
from app.schemas import DeleteResponse, DocumentInfoOut
from jera.rag import RagSystem

router = APIRouter(tags=["documents"])

SystemDep = Annotated[RagSystem, Depends(get_system)]


def _to_out(info: object) -> DocumentInfoOut:
    # info is a jera DocumentInfo (frozen pydantic) — map to the API shape.
    return DocumentInfoOut.model_validate(info, from_attributes=True)


@router.get("/documents", response_model=list[DocumentInfoOut])
def list_documents(system: SystemDep) -> list[DocumentInfoOut]:
    return [_to_out(d) for d in system.metadata_store.list_documents()]


@router.get("/documents/{document_id}", response_model=DocumentInfoOut)
def get_document(document_id: str, system: SystemDep) -> DocumentInfoOut:
    info = system.metadata_store.get_document_info(document_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"document {document_id!r} not found")
    return _to_out(info)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, system: SystemDep) -> DeleteResponse:
    if system.metadata_store.get_document_info(document_id) is None:
        raise HTTPException(status_code=404, detail=f"document {document_id!r} not found")
    deleted_chunk_ids = system.metadata_store.delete_document(document_id)
    if deleted_chunk_ids:
        system.vector_store.delete(system.collection, deleted_chunk_ids)
    return DeleteResponse(document_id=document_id, deleted_chunk_count=len(deleted_chunk_ids))
