"""FastAPI app factory — composition only (routers + DI). No domain logic lives here."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers import health, ingest, query


def create_app() -> FastAPI:
    app = FastAPI(title="Jera RAG API", version="0.1.0")
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    return app


app = create_app()
