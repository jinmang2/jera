"""PyHwpParser — Hancom legacy HWP (OLE binary) parser using the ``pyhwp`` package (``hwp5``).

HWP legacy format is an OLE2 Compound Binary File containing a ``PrvText`` stream
(UTF-16LE preview text) and a ``BodyText`` storage with per-section record streams.
We use the simplest stable surface:

    ``hwp5.filestructure.Hwp5File(path).preview_text.text``

``PrvText`` is the built-in quick-preview stream that Hancom embeds in every HWP v5
file — it contains the full document text as plain UTF-16LE with paragraph separators
(``\\r\\n``). This gives us clean paragraphs without requiring lxml/XSLT (the heavier
``hwp5txt`` path), and the API has been stable across pyhwp 0.1b8–0.1b15.

Source reference:
    https://github.com/mete0r/pyhwp  (``hwp5/filestructure.py``, ``Hwp5File.preview_text``)
    https://pyhwp.readthedocs.io/en/latest/hwp5.html

Because HWP files are OLE2 containers the library always expects a real filesystem
path. When the caller supplies in-memory bytes (``SourceRef.content``) we write them
to a ``NamedTemporaryFile`` first.

Honesty scope: paragraph-level text extraction only — no heading-style metadata,
no inline images/tables, no formulas.  Section paths use the same short-text
heuristic as HwpxParser.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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

# Paragraph separators written by Hancom into PrvText.
_PARA_SEPS = ("\r\n", "\r", "\n")


class PyHwpParser:
    """DocumentParser adapter for the Hancom legacy HWP binary format (OLE2/HWPv5).

    Requires the ``hwp`` optional extra (``pip install pyhwp``).
    Install: ``uv sync --extra hwp``
    """

    name = "pyhwp"
    version = "1.0.0"

    def supports(self, source: SourceRef) -> bool:
        """Return True only for legacy HWP (``application/x-hwp``)."""
        return source.media_type is MediaType.HWP

    def parse(self, source: SourceRef) -> ParsedDocument:
        """Parse a legacy ``.hwp`` file into a :class:`ParsedDocument`.

        Opens the OLE2 container via ``hwp5.filestructure.Hwp5File``, reads the
        embedded ``PrvText`` stream (``preview_text.text``), and splits it into
        paragraphs.  Each non-empty paragraph becomes a :class:`DocumentElement`
        typed as ``TITLE`` (short, no terminal punctuation) or ``NARRATIVE_TEXT``.
        """
        try:
            from hwp5.filestructure import Hwp5File, InvalidHwp5FileError
        except ImportError:  # pragma: no cover
            raise ImportError("PyHwpParser requires the 'hwp' extra: install pyhwp") from None

        document_id = stable_id(source.source_id, source.media_type.value, self.name)

        # hwp5 always needs a filesystem path; write bytes to a temp file when needed.
        if source.path is not None:
            hwp_path: str | Path = source.path
            raw_text = _open_and_read(hwp_path, Hwp5File, InvalidHwp5FileError)
        else:
            with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as tmp:
                tmp.write(source.read_bytes())
                tmp_path = tmp.name
            try:
                raw_text = _open_and_read(tmp_path, Hwp5File, InvalidHwp5FileError)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        elements: list[DocumentElement] = []
        title: str | None = None
        current_section: str | None = None
        order = 0

        for para in _split_paragraphs(raw_text):
            if not para:
                continue
            if _looks_like_heading(para):
                current_section = para
                if title is None:
                    title = para
                etype = ElementType.TITLE
                section_path: tuple[str, ...] = ()
            else:
                etype = ElementType.NARRATIVE_TEXT
                section_path = (current_section,) if current_section else ()

            elements.append(
                DocumentElement(
                    element_id=stable_id(document_id, str(order), etype.value),
                    type=etype,
                    text=para,
                    page_span=PageSpan.single(1),
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_and_read(path: str | Path, hwp5_file_cls: type, invalid_err: type) -> str:
    """Open an HWP file and return the preview text string."""
    hwp5file = hwp5_file_cls(str(path))
    return str(hwp5file.preview_text.text)


def _split_paragraphs(text: str) -> list[str]:
    """Split ``PrvText`` content on ``\\r\\n``, ``\\r``, or ``\\n``."""
    # Normalise to \n first, then split.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return [p.strip() for p in normalised.split("\n")]


def _looks_like_heading(text: str) -> bool:
    """Light heuristic (mirrors HwpxParser): short with no sentence-terminal punctuation."""
    return len(text) <= 40 and not text.rstrip().endswith((".", "!", "?"))
