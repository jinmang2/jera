"""Heading/element-aware Markdown + plain-text parser (pure Python, default).

Produces typed elements with section-path breadcrumbs and reading order. This is the M1
default for ``text/markdown`` and ``text/plain`` sources.
"""

from __future__ import annotations

import re

from jera.domain.document import (
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
    SourceRef,
)
from jera.domain.ids import stable_id

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+(.+)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_FENCE = re.compile(r"^\s*```")


class MarkdownParser:
    name = "markdown"
    version = "1.0.0"

    _SUPPORTED = {MediaType.MARKDOWN, MediaType.PLAIN}

    def supports(self, source: SourceRef) -> bool:
        return source.media_type in self._SUPPORTED

    def parse(self, source: SourceRef) -> ParsedDocument:
        text = source.read_text()
        document_id = stable_id(source.source_id, source.media_type.value, self.name)
        treat_headings = source.media_type is MediaType.MARKDOWN

        elements: list[DocumentElement] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        title: str | None = None
        order = 0
        lines = text.splitlines()
        i = 0

        def push(etype: ElementType, body: str, section_path: tuple[str, ...]) -> None:
            nonlocal order
            body = body.strip()
            if not body:
                return
            elements.append(
                DocumentElement(
                    element_id=stable_id(document_id, str(order), etype.value),
                    type=etype,
                    text=body,
                    page_span=PageSpan.single(1),
                    order=order,
                    section_path=section_path,
                )
            )
            order += 1

        while i < len(lines):
            line = lines[i]

            # Code fence: consume until closing fence.
            if treat_headings and _FENCE.match(line):
                j = i + 1
                buf: list[str] = []
                while j < len(lines) and not _FENCE.match(lines[j]):
                    buf.append(lines[j])
                    j += 1
                push(ElementType.CODE, "\n".join(buf), _section(heading_stack))
                i = j + 1
                continue

            # Heading.
            m = _HEADING.match(line) if treat_headings else None
            if m:
                level = len(m.group(1))
                htext = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                ancestors = _section(heading_stack)
                push(ElementType.TITLE, htext, ancestors)
                heading_stack.append((level, htext))
                if title is None and level == 1:
                    title = htext
                i += 1
                continue

            # Table: consecutive pipe rows.
            if treat_headings and _TABLE_ROW.match(line):
                j = i
                rows: list[str] = []
                while j < len(lines) and _TABLE_ROW.match(lines[j]):
                    rows.append(lines[j])
                    j += 1
                push(ElementType.TABLE, "\n".join(rows), _section(heading_stack))
                i = j
                continue

            # List item.
            lm = _LIST_ITEM.match(line) if treat_headings else None
            if lm:
                push(ElementType.LIST_ITEM, lm.group(2), _section(heading_stack))
                i += 1
                continue

            # Blank line: paragraph break.
            if not line.strip():
                i += 1
                continue

            # Paragraph: accumulate until blank/structural line.
            j = i
            para: list[str] = []
            while j < len(lines) and lines[j].strip():
                nxt = lines[j]
                if treat_headings and (
                    _HEADING.match(nxt)
                    or _TABLE_ROW.match(nxt)
                    or _LIST_ITEM.match(nxt)
                    or _FENCE.match(nxt)
                ):
                    break
                para.append(nxt)
                j += 1
            push(ElementType.NARRATIVE_TEXT, " ".join(para), _section(heading_stack))
            i = j

        return ParsedDocument(
            document_id=document_id,
            source_id=source.source_id,
            title=title,
            elements=elements,
            provenance=Provenance(
                source_id=source.source_id,
                parser_name=self.name,
                parser_version=self.version,
                media_type=source.media_type,
            ),
        )


def _section(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(t for _, t in stack)
