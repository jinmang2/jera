"""HeuristicContextualizer — deterministic Contextual Retrieval, no LLM, CI-real.

Builds each chunk's situating context from structure already present in the parsed document:
the document title and the chunk's heading breadcrumb (``section_path``). This is the cheap,
offline, language-neutral approximation of Anthropic's LLM-written context — it reliably adds
the document/section entities a chunk often omits, which is exactly what makes an otherwise
unfindable chunk retrievable. No vendor SDK, no network, fully deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument


class HeuristicContextualizer:
    """Situate each chunk with ``title`` + ``section_path`` (deterministic)."""

    strategy = "heuristic"
    version = "1.0"

    def contextualize(self, document: ParsedDocument, chunks: Sequence[Chunk]) -> list[str]:
        title = (document.title or "").strip()
        out: list[str] = []
        for chunk in chunks:
            parts: list[str] = []
            if title:
                parts.append(title)
            # section_path repeats the title as its first crumb for some parsers; de-dup it.
            for crumb in chunk.section_path:
                crumb = crumb.strip()
                if crumb and crumb not in parts:
                    parts.append(crumb)
            out.append(" › ".join(parts))
        return out
