"""Contextualizer adapters: heuristic determinism + LLM wiring (offline fake)."""

from __future__ import annotations

from jera.adapters.contextual.heuristic_contextualizer import HeuristicContextualizer
from jera.adapters.contextual.llm_contextualizer import LlmContextualizer
from jera.domain.chunk import Chunk
from jera.domain.document import (
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
)
from jera.ports.contextualizer import Contextualizer


def _doc(title: str | None, *texts: str) -> ParsedDocument:
    elements = [
        DocumentElement(
            element_id=f"e{i}",
            type=ElementType.NARRATIVE_TEXT,
            text=t,
            page_span=PageSpan.single(1),
            order=i,
        )
        for i, t in enumerate(texts)
    ]
    return ParsedDocument(
        document_id="d1",
        source_id="s1",
        title=title,
        elements=elements,
        provenance=Provenance(
            source_id="s1",
            parser_name="markdown",
            parser_version="1.0",
            media_type=MediaType.MARKDOWN,
        ),
    )


def _chunk(text: str, section_path: tuple[str, ...]) -> Chunk:
    return Chunk(
        chunk_id="c1",
        document_id="d1",
        source_id="s1",
        text=text,
        page_span=PageSpan.single(1),
        section_path=section_path,
        element_ids=("e0",),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="heading_aware",
        chunk_version="1.0",
    )


def test_heuristic_is_a_contextualizer() -> None:
    assert isinstance(HeuristicContextualizer(), Contextualizer)


def test_heuristic_combines_title_and_section_path() -> None:
    doc = _doc("Acme Report", "body")
    chunk = _chunk("Revenue grew.", ("Acme Report", "Financials", "Q4"))
    [ctx] = HeuristicContextualizer().contextualize(doc, [chunk])
    # Title is deduped against the repeated first breadcrumb; order preserved.
    assert ctx == "Acme Report › Financials › Q4"


def test_heuristic_is_deterministic() -> None:
    doc = _doc("T", "b")
    chunk = _chunk("x", ("T", "S"))
    ctxs = [HeuristicContextualizer().contextualize(doc, [chunk])[0] for _ in range(3)]
    assert len(set(ctxs)) == 1


def test_heuristic_handles_missing_title_and_sections() -> None:
    doc = _doc(None, "b")
    chunk = _chunk("x", ())
    [ctx] = HeuristicContextualizer().contextualize(doc, [chunk])
    assert ctx == ""  # nothing to situate with → empty, leaves chunk un-contextualized


def test_heuristic_returns_one_context_per_chunk() -> None:
    doc = _doc("T", "a", "b")
    chunks = [_chunk("a", ("T", "A")), _chunk("b", ("T", "B"))]
    out = HeuristicContextualizer().contextualize(doc, chunks)
    assert len(out) == len(chunks)


class _FakeSituateLLM:
    """Deterministic offline SituateLLM — proves LlmContextualizer wiring without a paid call."""

    model_id = "fake-situate"

    def situate(self, document_text: str, chunk_text: str) -> str:
        head = document_text.split("\n", 1)[0][:40]
        return f"[doc:{head}] mentions: {chunk_text.split()[0]}"


def test_llm_contextualizer_uses_injected_llm_over_whole_document() -> None:
    doc = _doc("Acme Report", "Intro paragraph.", "Revenue grew three percent.")
    chunk = _chunk("Revenue grew three percent.", ("Acme Report", "Financials"))
    ctxs = LlmContextualizer(_FakeSituateLLM()).contextualize(doc, [chunk])
    assert ctxs == ["[doc:Intro paragraph.] mentions: Revenue"]


def test_llm_contextualizer_is_a_contextualizer() -> None:
    assert isinstance(LlmContextualizer(_FakeSituateLLM()), Contextualizer)
