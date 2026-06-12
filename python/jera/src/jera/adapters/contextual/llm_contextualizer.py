"""LlmContextualizer — Anthropic-style LLM-written Contextual Retrieval (opt-in).

Anthropic's recipe ("Introducing Contextual Retrieval", 2024): for each chunk, prompt an LLM
with the *whole document* plus the chunk and ask for a short context that situates the chunk
within the document. With prompt caching the document is paid for once per ingest, so the
per-chunk cost is just the chunk + completion.

The LLM is injected behind the ``SituateLLM`` protocol so this adapter is **offline-testable**
with a deterministic fake (FakeToolUseLLM discipline). The real Claude implementation
(``ClaudeSituateLLM``) is constructed only with cloud enabled + a key; CI never builds it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument

# Anthropic's contextual-retrieval prompt (verbatim shape from the cookbook), split so a real
# adapter can send the document as a separate cacheable block. The two halves wrap the
# document/chunk text by concatenation (never str.format) so literal braces in a document can't
# break templating.
SITUATE_DOC_PREFIX = "<document>\n"
SITUATE_DOC_SUFFIX = (
    "\n</document>\nHere is the chunk we want to situate within the whole document:\n"
)
SITUATE_CHUNK_PREFIX = "<chunk>\n"
SITUATE_CHUNK_SUFFIX = (
    "\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)


def build_situate_prompt(document_text: str, chunk_text: str) -> str:
    """The full single-string situate prompt (for non-caching SituateLLM implementations)."""
    return (
        SITUATE_DOC_PREFIX
        + document_text
        + SITUATE_DOC_SUFFIX
        + SITUATE_CHUNK_PREFIX
        + chunk_text
        + SITUATE_CHUNK_SUFFIX
    )


@runtime_checkable
class SituateLLM(Protocol):
    """Turns (whole-document text, chunk text) into a short situating context string."""

    model_id: str

    def situate(self, document_text: str, chunk_text: str) -> str: ...


class LlmContextualizer:
    """Situate each chunk via an injected ``SituateLLM`` over the whole-document text."""

    strategy = "llm"
    version = "1.0"

    def __init__(self, llm: SituateLLM) -> None:
        self._llm = llm

    def contextualize(self, document: ParsedDocument, chunks: Sequence[Chunk]) -> list[str]:
        document_text = "\n\n".join(el.text for el in document.elements)
        return [self._llm.situate(document_text, chunk.text).strip() for chunk in chunks]
