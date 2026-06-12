"""CamelotTableParser — PDF table extractor using camelot-py.

Extracts tables from PDFs as structured TABLE elements (high value for
table-heavy Korean research corpora). Uses ``camelot.read_pdf`` with
configurable flavor (``"lattice"`` for ruled grids, ``"stream"`` for
whitespace-delimited); each extracted ``Table`` becomes one
``DocumentElement`` of type ``ElementType.TABLE`` whose ``.text`` is the
table rendered as pipe-delimited Markdown.

Camelot reads from a **file path** and requires Ghostscript (lattice mode).
When the ``SourceRef`` carries only in-memory bytes (``content``), the adapter
writes them to a temporary file for the duration of the call.

Install::

    pip install camelot-py[base]       # lattice (requires ghostscript)
    pip install camelot-py[cv]         # + stream / OpenCV extras

The ``tables`` optional-dependency group in this project pins camelot-py.
"""

from __future__ import annotations

import tempfile
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


def _df_to_markdown(df: Any) -> str:
    """Convert a camelot Table's DataFrame to a pipe-delimited Markdown string.

    Iterates rows/columns directly so that ``pandas`` is not required as a
    hard dependency in CI (the real ``df`` is a ``pd.DataFrame``; the fake
    used in SDK-boundary tests exposes only ``.iterrows()`` and ``.columns``).
    """
    columns: list[str] = [str(c) for c in df.columns]
    rows: list[list[str]] = [[str(v) for v in row] for _, row in df.iterrows()]

    header = " | ".join(columns)
    separator = " | ".join("---" for _ in columns)
    body_lines = [" | ".join(cells) for cells in rows]

    parts = [header, separator, *body_lines]
    return "\n".join(parts)


class CamelotTableParser:
    """Parse tables from a PDF using camelot-py.

    Parameters
    ----------
    flavor:
        Camelot parsing flavour — ``"lattice"`` (default, uses Ghostscript,
        suited for bordered grids) or ``"stream"`` (whitespace-based, no
        Ghostscript needed).
    pages:
        Comma-separated page selection string forwarded to
        ``camelot.read_pdf``, e.g. ``"all"``, ``"1,3-5"``.  Defaults to
        ``"all"`` so every page is scanned.
    suppress_stdout:
        Silence camelot progress output (default ``True``).
    """

    name = "camelot"
    version = "1.0.0"

    def __init__(
        self,
        flavor: str = "lattice",
        pages: str = "all",
        suppress_stdout: bool = True,
    ) -> None:
        self.flavor = flavor
        self.pages = pages
        self.suppress_stdout = suppress_stdout

    def supports(self, source: SourceRef) -> bool:
        return source.media_type is MediaType.PDF

    def parse(self, source: SourceRef) -> ParsedDocument:
        try:
            import camelot  # pragma: no cover
        except ImportError:  # pragma: no cover
            raise ImportError(
                "CamelotTableParser requires the 'tables' extra: install camelot-py[base]"
            ) from None

        document_id = stable_id(source.source_id, source.media_type.value, self.name)

        # camelot.read_pdf requires a filesystem path.
        if source.path is not None:
            filepath = str(source.path)
            tables = camelot.read_pdf(
                filepath,
                pages=self.pages,
                flavor=self.flavor,
                suppress_stdout=self.suppress_stdout,
            )
            elements = self._tables_to_elements(document_id, tables)
        else:
            # Write bytes to a temp file; cleaned up on context exit.
            data = source.read_bytes()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(data)
                tmp.flush()
                tables = camelot.read_pdf(
                    tmp.name,
                    pages=self.pages,
                    flavor=self.flavor,
                    suppress_stdout=self.suppress_stdout,
                )
                elements = self._tables_to_elements(document_id, tables)

        return ParsedDocument(
            document_id=document_id,
            source_id=source.source_id,
            title=None,
            elements=elements,
            provenance=Provenance(
                source_id=source.source_id,
                parser_name=self.name,
                parser_version=self.version,
                media_type=source.media_type,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tables_to_elements(self, document_id: str, tables: Any) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        for order, table in enumerate(tables):
            text = _df_to_markdown(table.df)
            if not text.strip():
                continue
            page_num: int = int(table.page)
            elements.append(
                DocumentElement(
                    element_id=stable_id(document_id, str(order), ElementType.TABLE.value),
                    type=ElementType.TABLE,
                    text=text,
                    page_span=PageSpan.single(page_num),
                    order=order,
                    section_path=(),
                )
            )
        return elements
