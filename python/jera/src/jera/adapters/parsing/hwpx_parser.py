"""HwpxParser — Hancom HWPX (OWPML) parser using ONLY stdlib (`zipfile` + ElementTree).

HWPX is a zip of OWPML XML: ``Contents/section*.xml`` hold the body as namespaced paragraphs
(``<hp:p><hp:run><hp:t>text``) and tables (``<hp:tbl><hp:tr><hp:tc>``). We match by *local*
tag name (namespace-agnostic) so minor OWPML namespace differences don't break parsing.

Honesty scope (cf. PyMuPDFParser): this proves the parser handles the OWPML element shape
(paragraphs + a flat/nested table) on a controlled fixture — it does NOT claim full fidelity
to every real Hancom document (style-driven headings, complex nested tables, equations,
text-boxes are out of M5a scope). Section paths use a light heading heuristic, not OWPML styles.
No lxml: `lxml` is not a base dependency (only reachable via the docling extra).
"""

from __future__ import annotations

import re
import zipfile
from xml.etree import ElementTree as ET

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

_SECTION_RE = re.compile(r"(?:^|/)section(\d+)\.xml$", re.IGNORECASE)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class HwpxParser:
    name = "hwpx"
    version = "1.0.0"

    def supports(self, source: SourceRef) -> bool:
        return source.media_type is MediaType.HWPX

    def parse(self, source: SourceRef) -> ParsedDocument:
        document_id = stable_id(source.source_id, source.media_type.value, self.name)
        elements: list[DocumentElement] = []
        title: str | None = None
        order = 0
        current_section: str | None = None

        import io

        with zipfile.ZipFile(io.BytesIO(source.read_bytes())) as zf:
            section_names = sorted(
                (n for n in zf.namelist() if _SECTION_RE.search(n)),
                key=lambda n: int(_SECTION_RE.search(n).group(1)),  # type: ignore[union-attr]
            )
            for name in section_names:
                root = ET.fromstring(zf.read(name))
                # Collect the descendant set of every table so paragraph text excludes table cells.
                in_table: set[int] = set()
                for tbl in (e for e in root.iter() if _local(e.tag) == "tbl"):
                    for d in tbl.iter():
                        in_table.add(id(d))

                for el in root.iter():
                    loc = _local(el.tag)
                    if loc == "tbl":
                        text = _table_text(el)
                        if text:
                            elements.append(
                                _make(document_id, order, ElementType.TABLE, text, current_section)
                            )
                            order += 1
                    elif loc == "p" and id(el) not in in_table:
                        text = _para_text(el, in_table)
                        if not text:
                            continue
                        if _looks_like_heading(text):
                            current_section = text
                            if title is None:
                                title = text
                            elements.append(
                                _make(document_id, order, ElementType.TITLE, text, None)
                            )
                        else:
                            elements.append(
                                _make(
                                    document_id,
                                    order,
                                    ElementType.NARRATIVE_TEXT,
                                    text,
                                    current_section,
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


def _para_text(p: ET.Element, in_table: set[int]) -> str:
    parts = [t.text for t in p.iter() if _local(t.tag) == "t" and id(t) not in in_table and t.text]
    return " ".join(s.strip() for s in parts if s.strip()).strip()


def _table_text(tbl: ET.Element) -> str:
    rows: list[str] = []
    for tr in (e for e in tbl.iter() if _local(e.tag) == "tr"):
        cells = [
            " ".join(t.text.strip() for t in tc.iter() if _local(t.tag) == "t" and t.text)
            for tc in (c for c in tr.iter() if _local(c.tag) == "tc")
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _looks_like_heading(text: str) -> bool:
    # Light heuristic (not OWPML styles): short, with no sentence-terminal punctuation.
    # (Bare Korean endings like 다/요 are NOT used — they false-match nouns such as 개요/요약.)
    return len(text) <= 40 and not text.rstrip().endswith((".", "!", "?"))


def _make(
    document_id: str,
    order: int,
    etype: ElementType,
    text: str,
    section: str | None,
) -> DocumentElement:
    return DocumentElement(
        element_id=stable_id(document_id, str(order), etype.value),
        type=etype,
        text=text,
        page_span=PageSpan.single(1),
        order=order,
        section_path=(section,) if section else (),
    )
