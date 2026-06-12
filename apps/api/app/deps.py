"""Dependency injection — builds the RagSystem from settings and shares it per process."""

from __future__ import annotations

from functools import lru_cache

from jera.rag import RagSystem, Settings, build_system


@lru_cache(maxsize=1)
def get_system() -> RagSystem:
    """Build (once) and return the wired RAG system for the active profile."""
    return build_system(Settings())
