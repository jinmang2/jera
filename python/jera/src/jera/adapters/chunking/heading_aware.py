"""Heading/structure-aware chunker (M1 baseline).

Groups consecutive elements into sections keyed by their heading breadcrumb, concatenates
each section's text, and windows it into token-bounded chunks. Deterministic: identical
input always yields identical chunk ids, section paths, page spans, and char spans.
"""

from __future__ import annotations

from jera.adapters.chunking.tokenizer import count_tokens
from jera.domain.chunk import Chunk
from jera.domain.document import DocumentElement, PageSpan, ParsedDocument
from jera.domain.ids import stable_id


def _section_key(el: DocumentElement) -> tuple[str, ...]:
    # A title starts a new section equal to its breadcrumb + own text; content inherits
    # the breadcrumb it already carries.
    if el.type.value == "Title":
        return (*el.section_path, el.text)
    return el.section_path


class HeadingAwareChunker:
    strategy = "heading_aware"
    version = "1.0.0"

    def __init__(self, max_tokens: int = 180, overlap_tokens: int = 40) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in self._group_sections(document.elements):
            chunks.extend(self._chunk_section(document, section))
        return chunks

    def _group_sections(self, elements: list[DocumentElement]) -> list[list[DocumentElement]]:
        groups: list[list[DocumentElement]] = []
        current_key: tuple[str, ...] | None = None
        for el in elements:
            key = _section_key(el)
            if key != current_key:
                groups.append([])
                current_key = key
            groups[-1].append(el)
        return groups

    def _chunk_section(
        self, document: ParsedDocument, section: list[DocumentElement]
    ) -> list[Chunk]:
        if not section:
            return []
        section_path = _section_key(section[0])
        # Concatenate element texts, recording each element's [start, end) char range.
        parts: list[str] = []
        ranges: list[tuple[int, int, str, PageSpan]] = []  # (start, end, element_id, page_span)
        cursor = 0
        for el in section:
            if parts:
                cursor += 2  # the "\n\n" join inserted before this element
            ranges.append((cursor, cursor + len(el.text), el.element_id, el.page_span))
            parts.append(el.text)
            cursor += len(el.text)
        section_text = "\n\n".join(parts)

        # Window over tokens, mapping token windows back to char offsets.
        tokens_with_offsets = _tokenize_with_offsets(section_text)
        if not tokens_with_offsets:
            return []

        out: list[Chunk] = []
        step = self.max_tokens - self.overlap_tokens
        for window_idx, start in enumerate(range(0, len(tokens_with_offsets), step)):
            window = tokens_with_offsets[start : start + self.max_tokens]
            if not window:
                break
            char_start = window[0][1]
            char_end = window[-1][2]
            text = section_text[char_start:char_end]
            element_ids = tuple(
                eid
                for (e_start, e_end, eid, _) in ranges
                if e_start < char_end and char_start < e_end  # ranges overlap
            )
            page_span = _merge_pages(section, element_ids) or section[0].page_span
            chunk_id = stable_id(
                document.document_id,
                self.strategy,
                self.version,
                "/".join(section_path),
                str(window_idx),
                str(char_start),
            )
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_id=document.source_id,
                    text=text,
                    page_span=page_span,
                    section_path=section_path,
                    element_ids=element_ids or tuple(e.element_id for e in section),
                    char_span=(char_start, char_end),
                    token_count=count_tokens(text),
                    chunk_strategy=self.strategy,
                    chunk_version=self.version,
                    parent_chunk_id=None,
                )
            )
            if start + self.max_tokens >= len(tokens_with_offsets):
                break
        return out


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return (token, start, end) char offsets for whitespace tokens."""
    out: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        out.append((text[start:i], start, i))
    return out


def _merge_pages(section: list[DocumentElement], element_ids: tuple[str, ...]) -> PageSpan | None:
    spans = [e.page_span for e in section if e.element_id in element_ids]
    if not spans:
        return None
    merged = spans[0]
    for s in spans[1:]:
        merged = merged.merge(s)
    return merged
