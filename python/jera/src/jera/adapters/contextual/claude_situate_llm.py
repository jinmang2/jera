"""ClaudeSituateLLM — Anthropic SituateLLM for Contextual Retrieval. DISABLED by default.

Implements the ``SituateLLM`` port with a real Claude call. The whole-document block is sent
with a ``cache_control`` breakpoint so Anthropic prompt caching charges for it once per ingest
(the per-chunk cost is then just the chunk + the short completion) — this is what makes
LLM-written contextual retrieval affordable. Never enabled in automated tests.
"""

from __future__ import annotations

from jera.adapters.contextual.llm_contextualizer import (
    SITUATE_CHUNK_PREFIX,
    SITUATE_CHUNK_SUFFIX,
    SITUATE_DOC_PREFIX,
    SITUATE_DOC_SUFFIX,
)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap + fast; situating is a small task


class ClaudeSituateLLM:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        enabled: bool = False,
        max_tokens: int = 256,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeSituateLLM is disabled by default. Pass enabled=True and an api_key "
                "(paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("ClaudeSituateLLM requires an api_key when enabled.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeSituateLLM requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model_id = model
        self._max_tokens = max_tokens

    def situate(self, document_text: str, chunk_text: str) -> str:  # pragma: no cover
        # Send the document (cacheable, identical across every chunk of the same document) as
        # its own content block with a cache breakpoint; the chunk block varies per call.
        doc_block = SITUATE_DOC_PREFIX + document_text + SITUATE_DOC_SUFFIX
        chunk_block = SITUATE_CHUNK_PREFIX + chunk_text + SITUATE_CHUNK_SUFFIX
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": doc_block,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": chunk_block},
                    ],
                }
            ],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
