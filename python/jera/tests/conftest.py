"""Shared fixtures: sample documents, a generated text PDF, and a fresh test RAG system."""

from __future__ import annotations

import pytest

from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings

SAMPLE_MARKDOWN = """# Jera Overview

Jera is a hexagonal, offline-first retrieval system. The identifier ZX9000 names the ranking module.

## Retrieval

It supports dense, sparse, and hybrid retrieval.
Hybrid uses reciprocal rank fusion to merge dense and sparse rankings.

## Storage

Postgres owns documents and chunks. Qdrant owns dense and sparse named vectors.
"""

SAMPLE_TABLE_MARKDOWN = """# Benchmarks

This section has a table of results.

| Strategy | Recall | Notes |
| --- | --- | --- |
| dense | 0.71 | semantic recall |
| sparse | 0.65 | exact terms |
| hybrid | 0.80 | fused |

The hybrid row wins overall.
"""


def make_text_pdf(lines: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "\n".join(lines), fontsize=12)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


@pytest.fixture
def sample_markdown() -> str:
    return SAMPLE_MARKDOWN


@pytest.fixture
def sample_table_markdown() -> str:
    return SAMPLE_TABLE_MARKDOWN


@pytest.fixture
def text_pdf_bytes() -> bytes:
    return make_text_pdf(
        [
            "Jera Technical Note",
            "Reciprocal rank fusion combines dense and sparse rankings.",
            "The ranking module identifier is ZX9000.",
        ]
    )


@pytest.fixture
def system() -> RagSystem:
    """A fresh, fully deterministic test-profile system (in-memory store + SQLite memory)."""
    return build_system(Settings(profile=Profile.TEST))
