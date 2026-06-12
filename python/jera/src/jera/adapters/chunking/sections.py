"""Group parsed elements into heading-keyed sections with char-offset provenance.

Shared by the heading-aware and semantic chunkers so section grouping and element/page
attribution are defined once.
"""

from __future__ import annotations

from dataclasses import dataclass

from jera.domain.document import DocumentElement, PageSpan


def section_key(el: DocumentElement) -> tuple[str, ...]:
    """A title starts a new section (breadcrumb + own text); content inherits its breadcrumb."""
    if el.type.value == "Title":
        return (*el.section_path, el.text)
    return el.section_path


@dataclass
class Section:
    section_path: tuple[str, ...]
    text: str
    # (start, end, element_id, page_span) for each contributing element within `text`
    ranges: list[tuple[int, int, str, PageSpan]]

    def attribute(self, char_start: int, char_end: int) -> tuple[tuple[str, ...], PageSpan]:
        """Return the element_ids and merged page span overlapping [char_start, char_end)."""
        ids: list[str] = []
        spans: list[PageSpan] = []
        for e_start, e_end, eid, page in self.ranges:
            if e_start < char_end and char_start < e_end:  # ranges overlap
                ids.append(eid)
                spans.append(page)
        if not spans:
            # fall back to whole section
            ids = [r[2] for r in self.ranges]
            spans = [r[3] for r in self.ranges]
        merged = spans[0]
        for s in spans[1:]:
            merged = merged.merge(s)
        return tuple(ids), merged


def group_sections(elements: list[DocumentElement]) -> list[Section]:
    sections: list[Section] = []
    current_key: tuple[str, ...] | None = None
    buf: list[DocumentElement] = []

    def flush() -> None:
        if not buf:
            return
        parts: list[str] = []
        ranges: list[tuple[int, int, str, PageSpan]] = []
        cursor = 0
        for el in buf:
            if parts:
                cursor += 2  # the "\n\n" join
            ranges.append((cursor, cursor + len(el.text), el.element_id, el.page_span))
            parts.append(el.text)
            cursor += len(el.text)
        sections.append(
            Section(section_path=section_key(buf[0]), text="\n\n".join(parts), ranges=ranges)
        )

    for el in elements:
        key = section_key(el)
        if key != current_key:
            flush()
            buf = []
            current_key = key
        buf.append(el)
    flush()
    return sections
