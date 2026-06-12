"""ClaudeHypothesisLLM — Anthropic HypothesisLLM for HyDE. DISABLED by default (paid).

Writes a short hypothetical answer passage used as an extra retrieval query. Never enabled in
automated tests; constructed only with cloud enabled + an api key.
"""

from __future__ import annotations

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap + fast; a one-paragraph hypothesis

_PROMPT = (
    "Write a short, factual passage (2-3 sentences) that directly answers the question below, "
    "as if quoted from a reference document. Do not hedge or mention that it is hypothetical.\n\n"
    "Question: "
)


class ClaudeHypothesisLLM:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        enabled: bool = False,
        max_tokens: int = 256,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeHypothesisLLM is disabled by default. Pass enabled=True and an api_key "
                "(paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("ClaudeHypothesisLLM requires an api_key when enabled.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeHypothesisLLM requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model_id = model
        self._max_tokens = max_tokens

    def hypothesize(self, query: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": _PROMPT + query}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
