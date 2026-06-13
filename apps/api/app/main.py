"""FastAPI app factory — composition only (routers + DI). No domain logic lives here."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers import documents, health, ingest, jobs, query


def create_app() -> FastAPI:
    app = FastAPI(title="Jera RAG API", version="0.1.0")
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(jobs.router)
    app.include_router(documents.router)
    return app


app = create_app()
