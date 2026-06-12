"""OpenDataLoader PDF parser adapter (behind the ``opendataloader`` extra).

Converts PDF via `opendataloader-pdf` (Apache-2.0, Java 11+ required) and maps
its JSON layout elements into Jera's typed element model with section-path
breadcrumbs, page spans, and routing provenance.

``opendataloader_pdf.convert()`` is a write-to-disk API: it accepts a list of
input paths plus an output directory and produces ``<stem>.json`` files for each
input.  This adapter materialises in-memory bytes to a temporary file, invokes
the converter, reads back the JSON result, then removes both the temp input and
output before returning.

Lazy imports keep the base install light.  Java 11+ must be on PATH; the pip
package is a thin wrapper around the upstream Java CLI.

JSON element schema (per element)::

    {
        "type": "heading|paragraph|table|image|formula|chart",
        "id": <int>,
        "page_number": <int>,                        # 1-indexed
        "bounding_box": [left, bottom, right, top],  # PDF points
        "heading_level": <int | null>,               # 1+ for headings
        "font": "<str>",
        "font_size": <float>,
        "text_color": "<str>",
        "content": "<str>"
    }
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from jera.domain.document import (
    METADATA_OCR_ENGINE,
    METADATA_ROUTE,
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
    SourceRef,
)
from jera.domain.ids import stable_id

# opendataloader element type string → Jera ElementType.
_TYPE_MAP: dict[str, ElementType] = {
    "heading": ElementType.TITLE,
    "paragraph": ElementType.NARRATIVE_TEXT,
    "table": ElementType.TABLE,
    "image": ElementType.FIGURE,
    "picture": ElementType.FIGURE,
    "chart": ElementType.FIGURE,
    "formula": ElementType.FORMULA,
    "list": ElementType.LIST_ITEM,
    "caption": ElementType.NARRATIVE_TEXT,
    "footnote": ElementType.NARRATIVE_TEXT,
    "page_header": ElementType.PAGE_HEADER,
    "page_footer": ElementType.PAGE_FOOTER,
}
_HEADING_TYPES = {"heading"}


class OpenDataLoaderParser:
    """PDF parser backed by ``opendataloader-pdf`` (Java 11+, layout+OCR+table)."""

    name = "opendataloader"
    version = "1.0.0"

    _SUPPORTED = {MediaType.PDF}

    def supports(self, source: SourceRef) -> bool:
        return source.media_type in self._SUPPORTED

    def parse(self, source: SourceRef) -> ParsedDocument:
        # Lazy import (like DoclingParser): constructing the parser must not require the extra,
        # so the registry can register it; only actual parsing needs the lib + Java 11+.
        try:
            import opendataloader_pdf as _odl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenDataLoaderParser requires the 'opendataloader' extra: "
                "`uv sync --extra opendataloader`. "
                "The underlying tool also needs Java 11+ on PATH."
            ) from exc

        document_id = stable_id(source.source_id, source.media_type.value, self.name)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / f"{source.source_id}.pdf"
            pdf_path.write_bytes(source.read_bytes())

            _odl.convert(
                input_path=[str(pdf_path)],
                output_dir=str(tmp_path),
                format="json",
            )

            json_path = tmp_path / f"{source.source_id}.json"
            raw: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))

        elements: list[DocumentElement] = []
        heading_stack: list[tuple[int, str]] = []  # (heading_level, text)
        title: str | None = None
        order = 0

        for item in raw:
            raw_type = str(item.get("type", "paragraph")).lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            etype = _TYPE_MAP.get(raw_type, ElementType.NARRATIVE_TEXT)
            page_no: int = int(item.get("page_number", 1))
            heading_level: int = int(item.get("heading_level") or 1)

            if raw_type in _HEADING_TYPES:
                while heading_stack and heading_stack[-1][0] >= heading_level:
                    heading_stack.pop()
                section_path = tuple(t for _, t in heading_stack)
                heading_stack.append((heading_level, content))
                if title is None:
                    title = content
            else:
                section_path = tuple(t for _, t in heading_stack)

            metadata: dict[str, object] = {
                METADATA_ROUTE: "text",
            }
            if raw_type == "image":
                metadata[METADATA_ROUTE] = "vlm"
            bbox = item.get("bounding_box")
            if bbox is not None:
                metadata["bounding_box"] = bbox
            ocr_engine = item.get("ocr_engine")
            if ocr_engine:
                metadata[METADATA_ROUTE] = "ocr"
                metadata[METADATA_OCR_ENGINE] = str(ocr_engine)

            elements.append(
                DocumentElement(
                    element_id=stable_id(document_id, str(order), etype.value),
                    type=etype,
                    text=content,
                    page_span=PageSpan.single(page_no),
                    order=order,
                    section_path=section_path,
                    metadata=metadata,
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
