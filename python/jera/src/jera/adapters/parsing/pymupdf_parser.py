"""PyMuPDF-based PDF parser (default for application/pdf in M1).

Extracts text blocks per page with reading order and page spans. Heading detection is a
light heuristic (short, larger-font blocks); table/figure semantics are intentionally NOT
claimed here — promote to Docling if fixture table/scan fidelity is insufficient (see ADR).
"""

from __future__ import annotations

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


class PyMuPDFParser:
    name = "pymupdf"
    version = "1.0.0"

    def supports(self, source: SourceRef) -> bool:
        return source.media_type is MediaType.PDF

    def parse(self, source: SourceRef) -> ParsedDocument:
        import pymupdf  # lazy import; base dependency

        # PyMuPDF ships no type information; treat the module as untyped (Any) at the boundary.
        pdf: Any = pymupdf
        document_id = stable_id(source.source_id, source.media_type.value, self.name)
        data = source.read_bytes()
        doc = pdf.open(stream=data, filetype="pdf")
        elements: list[DocumentElement] = []
        title: str | None = None
        order = 0
        try:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                # blocks: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                for b in blocks:
                    body = (b[4] or "").strip()
                    if not body:
                        continue
                    etype = ElementType.NARRATIVE_TEXT
                    # Heuristic: a short single-line block near the top is a heading.
                    if len(body) <= 80 and "\n" not in body and b[1] < page.rect.height * 0.25:
                        etype = ElementType.TITLE
                        if title is None and page_index == 0:
                            title = body
                    elements.append(
                        DocumentElement(
                            element_id=stable_id(document_id, str(order), etype.value),
                            type=etype,
                            text=body,
                            page_span=PageSpan.single(page_index + 1),
                            order=order,
                            section_path=(),
                        )
                    )
                    order += 1
        finally:
            doc.close()

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
