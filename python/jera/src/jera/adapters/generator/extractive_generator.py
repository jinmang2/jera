"""Extractive generator — TEST default.

Deterministic: builds an answer by stitching the top context snippets and emits one citation
per context chunk. No model call, so the E2E test is reproducible. Citations resolve to the
exact chunks passed in (the query pipeline guarantees those came from retrieval).
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.answer import Answer, Citation
from jera.domain.chunk import Chunk

_SNIPPET_CHARS = 240


class ExtractiveGenerator:
    model_id = "extractive-v1"

    def __init__(self, max_contexts: int = 3) -> None:
        self._max_contexts = max_contexts

    def generate(self, query: str, contexts: Sequence[Chunk]) -> Answer:
        used = list(contexts)[: self._max_contexts]
        if not used:
            return Answer(query=query, text="", citations=[])
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                snippet=_snippet(c.text),
                score=0.0,
                page_span=(c.page_span.start_page, c.page_span.end_page),
                section_path=c.section_path,
            )
            for c in used
        ]
        body = "\n\n".join(f"[{i + 1}] {_snippet(c.text)}" for i, c in enumerate(used))
        text = f"Based on {len(used)} retrieved passage(s):\n\n{body}"
        return Answer(query=query, text=text, citations=citations)


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[:_SNIPPET_CHARS].rstrip() + "…"
