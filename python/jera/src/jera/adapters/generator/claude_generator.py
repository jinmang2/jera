"""Anthropic Claude generator — DISABLED by default (extra: cloud, paid).

The recommended generation/long-context candidate. Builds a grounded prompt from retrieved
chunks and asks the model to answer with citations. Never enabled in automated tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.answer import Answer, Citation
from jera.domain.chunk import Chunk

_DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeGenerator:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        enabled: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeGenerator is disabled by default. Pass enabled=True and an api_key "
                "(paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("ClaudeGenerator requires an api_key when enabled.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeGenerator requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model_id = model
        self._max_tokens = max_tokens

    def generate(self, query: str, contexts: Sequence[Chunk]) -> Answer:  # pragma: no cover
        context_block = "\n\n".join(
            f"[{i + 1}] (chunk {c.chunk_id}) {c.text}" for i, c in enumerate(contexts)
        )
        prompt = (
            "Answer the question using ONLY the numbered context passages. "
            "Cite passages by their [n] markers.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                snippet=c.text[:240],
                score=0.0,
                page_span=(c.page_span.start_page, c.page_span.end_page),
                section_path=c.section_path,
            )
            for c in contexts
        ]
        return Answer(query=query, text=text, citations=citations)
