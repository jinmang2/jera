"""Proposition-based chunker — atomic retrieval units (M12).

Research basis
--------------
Chen et al., "Dense X Retrieval: What Retrieval Granularity Should We Use?",
EMNLP 2024.  arXiv:2312.06648 / ACL Anthology: 2024.emnlp-main.845

Key insight (§3 of the M12 research note)
------------------------------------------
Retrieval precision improves when every indexed unit is *atomic* — one fact per
chunk — so that a query about fact A does not drag in an unrelated fact B that
happens to live in the same passage.  The full paper accomplishes this with a
Flan-T5-large "Propositionizer" (not offline-deterministic).  Jera's port is a
deterministic offline approximation: **one sentence = one proposition**.

Self-containment without an LLM (Contextual Retrieval pattern)
---------------------------------------------------------------
A bare sentence often lacks its subject or topic (e.g. "It is 330 metres tall.")
To make each unit self-contained for dense retrieval, the section heading
breadcrumb is prepended as context — exactly the Contextual Retrieval mechanism
(Anthropic, 2024) already supported by ``Chunk.context`` / ``Chunk.embedding_text``.

* ``Chunk.text``           = the raw sentence (used for citations / snippets)
* ``Chunk.context``        = the heading breadcrumb prefix
* ``Chunk.embedding_text`` = ``context + "\\n\\n" + text``  (property, auto-derived)

This means the indexed/embedded form is heading-prefixed and self-contained,
while provenance (char_span, element_ids, page_span, section_path) is unaffected.
"""

from __future__ import annotations

from jera.adapters.chunking.sections import group_sections
from jera.adapters.chunking.sentences import split_sentences_with_offsets
from jera.adapters.chunking.tokenizer import count_tokens
from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument
from jera.domain.ids import stable_id


class PropositionChunker:
    """Split each section into individual sentences, one ``Chunk`` per sentence.

    Each chunk is an *atomic retrieval unit*: it contains exactly one sentence
    so that retrieving it does not pull in unrelated facts from the same passage.
    The heading breadcrumb is stored in ``Chunk.context`` and surfaced through
    ``Chunk.embedding_text`` to make each unit self-contained during dense
    retrieval, without altering ``Chunk.text`` or any provenance fields.

    Parameters
    ----------
    skip_single_word:
        Drop sentence fragments that are a single whitespace-delimited token
        (e.g. orphaned bullets).  Default ``True``.
    """

    strategy = "proposition"
    version = "1.0.0"

    def __init__(self, *, skip_single_word: bool = True) -> None:
        self.skip_single_word = skip_single_word

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        out: list[Chunk] = []
        for section in group_sections(document.elements):
            for sent_idx, (sent_text, s_start, s_end) in enumerate(
                split_sentences_with_offsets(section.text)
            ):
                if self.skip_single_word and len(sent_text.split()) <= 1:
                    continue

                element_ids, page_span = section.attribute(s_start, s_end)

                # Heading breadcrumb — e.g. ("Introduction", "Goals") → "Introduction > Goals"
                heading_prefix = " > ".join(section.section_path) if section.section_path else ""

                chunk_id = stable_id(
                    document.document_id,
                    self.strategy,
                    self.version,
                    "/".join(section.section_path),
                    str(sent_idx),
                    str(s_start),
                )

                out.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        source_id=document.source_id,
                        text=sent_text,
                        page_span=page_span,
                        section_path=section.section_path,
                        element_ids=element_ids,
                        char_span=(s_start, s_end),
                        token_count=count_tokens(sent_text),
                        chunk_strategy=self.strategy,
                        chunk_version=self.version,
                        parent_chunk_id=None,
                        # Self-containment prefix: makes embedding_text heading-scoped.
                        # Empty string → no prefix (no section heading in document).
                        context=heading_prefix if heading_prefix else None,
                    )
                )
        return out
