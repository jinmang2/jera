"""Docling parser adapter (behind the `docling` extra).

Converts PDF/HTML/Markdown via Docling and maps its `DoclingDocument` into Jera's typed
element model with section-path breadcrumbs, page spans, and provenance. Docling adds layout
analysis, table-structure recognition, and OCR for scanned PDFs — promote it over PyMuPDF
when fixture table/scan fidelity demands it (see ADR).

Lazy imports keep the base install light; the converter (which may download OCR models on
first use) is constructed once per parser instance.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

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

# Docling label (value) → Jera element type.
_LABEL_MAP = {
    "title": ElementType.TITLE,
    "section_header": ElementType.TITLE,
    "text": ElementType.NARRATIVE_TEXT,
    "paragraph": ElementType.NARRATIVE_TEXT,
    "caption": ElementType.NARRATIVE_TEXT,
    "footnote": ElementType.NARRATIVE_TEXT,
    "list_item": ElementType.LIST_ITEM,
    "table": ElementType.TABLE,
    "picture": ElementType.FIGURE,
    "chart": ElementType.FIGURE,
    "code": ElementType.CODE,
    "formula": ElementType.FORMULA,
    "page_header": ElementType.PAGE_HEADER,
    "page_footer": ElementType.PAGE_FOOTER,
}
_HEADINGS = {"title", "section_header"}
_EXT = {MediaType.PDF: "pdf", MediaType.HTML: "html", MediaType.MARKDOWN: "md"}


class DoclingParser:
    name = "docling"
    version = "1.0.0"

    _SUPPORTED = {MediaType.PDF, MediaType.HTML, MediaType.MARKDOWN}

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "DoclingParser requires the 'docling' extra: `uv sync --extra docling`."
            ) from exc
        self._converter = DocumentConverter()

    def supports(self, source: SourceRef) -> bool:
        return source.media_type in self._SUPPORTED

    def parse(self, source: SourceRef) -> ParsedDocument:
        from docling.datamodel.base_models import DocumentStream

        document_id = stable_id(source.source_id, source.media_type.value, self.name)
        ext = _EXT[source.media_type]
        stream = DocumentStream(
            name=f"{source.source_id}.{ext}", stream=BytesIO(source.read_bytes())
        )
        dl_doc = self._converter.convert(stream).document

        elements: list[DocumentElement] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        title: str | None = None
        order = 0

        for item, level in dl_doc.iterate_items():
            label = getattr(getattr(item, "label", None), "value", None)
            if label is None:
                continue
            etype = _LABEL_MAP.get(label, ElementType.NARRATIVE_TEXT)
            text = _item_text(item, dl_doc)
            if not text.strip():
                continue

            if label in _HEADINGS:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                section_path = tuple(t for _, t in heading_stack)
                heading_stack.append((level, text))
                if title is None:
                    title = text
            else:
                section_path = tuple(t for _, t in heading_stack)

            elements.append(
                DocumentElement(
                    element_id=stable_id(document_id, str(order), etype.value),
                    type=etype,
                    text=text,
                    page_span=_page_span(item),
                    order=order,
                    section_path=section_path,
                )
            )
            order += 1

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


def _item_text(item: Any, dl_doc: Any) -> str:
    text = getattr(item, "text", None)
    if text:
        return str(text)
    # Tables/pictures have no `.text`; export tables to markdown for retrievable content.
    exporter = getattr(item, "export_to_markdown", None)
    if exporter is not None:
        try:
            return str(exporter(dl_doc))
        except TypeError:  # older signature without the doc argument
            return str(exporter())
    return ""


def _page_span(item: Any) -> PageSpan:
    prov = getattr(item, "prov", None)
    if prov:
        pages = [p.page_no for p in prov if getattr(p, "page_no", None) is not None]
        if pages:
            return PageSpan(start_page=min(pages), end_page=max(pages))
    return PageSpan.single(1)
