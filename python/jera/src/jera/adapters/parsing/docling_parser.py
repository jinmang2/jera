"""Docling parser adapter (candidate; behind the `docling` extra).

Not the M1 default — promoted only if PyMuPDF table/scan fidelity proves insufficient on
fixture benchmarks (see ADR). Implements the same DocumentParser port. Importing Docling is
lazy so the base install stays light; without the extra, construction raises a clear error.
"""

from __future__ import annotations

from jera.domain.document import MediaType, ParsedDocument, SourceRef


class DoclingParser:
    name = "docling"
    version = "0.1.0"

    _SUPPORTED = {MediaType.PDF, MediaType.HTML, MediaType.MARKDOWN}

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only with extra absent
            raise ImportError(
                "DoclingParser requires the 'docling' extra: `uv sync --extra docling`."
            ) from exc

    def supports(self, source: SourceRef) -> bool:
        return source.media_type in self._SUPPORTED

    def parse(self, source: SourceRef) -> ParsedDocument:  # pragma: no cover - requires extra
        raise NotImplementedError(
            "DoclingParser.parse is contract-only until the Docling milestone; "
            "the port and dispatch are wired, the conversion mapping is not yet implemented."
        )
