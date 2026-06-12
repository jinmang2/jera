"""Contextual Retrieval (Anthropic, 2024) — NON-TAUTOLOGICAL lift, offline/deterministic.

Proves the feature does real work end-to-end through `build_system` + the ingest/query
pipelines. The answer lives in the `acme-annual` *content* chunk, which says "outlook" but never
names "Acme". Three rival companies publish their own "outlook" chunks. WITHOUT context the
answer chunk loses to those rivals (it matches only the common term "outlook"); WITH Contextual
Retrieval the document title supplies the rare term "acme" to the chunk's index text, so it
outranks every rival. The lift is idf-driven, not rigged — we compare the answer chunk's rank
against the rival outlook chunks, not against trivially-short title chunks.
"""

from __future__ import annotations

from jera.config.registry import build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import Query, RetrievalMode

# The target's chunk text deliberately omits "Acme" — only the H1 title carries the company.
CORPUS = {
    "acme-annual": (
        "# Acme Corporation Annual Report\n\n"
        "## Management Discussion\n\n"
        "The outlook for the coming year is positive, with sustained expansion "
        "anticipated across all segments.\n"
    ),
    # Mentions "Acme" but never "outlook" — the high-idf decoy that wins WITHOUT context.
    "acme-press": (
        "# Acme Press Release\n\n"
        "## Contact\n\n"
        "For media inquiries about Acme, please contact the press office directly.\n"
    ),
    # Mention "outlook" but never "Acme".
    "globex": (
        "# Globex Briefing\n\n## Summary\n\nThe outlook here is uncertain given headwinds.\n"
    ),
    "initech": (
        "# Initech Memo\n\n## Notes\n\nOur outlook remains cautious for the period ahead.\n"
    ),
}

QUERY = "Acme outlook"


def _ranked_outlook_sources(*, contextual: bool) -> list[str]:
    """Ranked source_ids of the *content* chunks (those whose text mentions 'outlook'),
    dropping trivially-short title-only chunks so we measure the real retrieval contest."""
    settings = Settings(profile=Profile.TEST, use_contextual_retrieval=contextual)
    system = build_system(settings)
    sources = [
        SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
        for sid, md in CORPUS.items()
    ]
    system.ingest.ingest_many(sources)
    result = system.query.retrieve(Query(text=QUERY, top_k=20, mode=RetrievalMode.SPARSE))
    ranked: list[str] = []
    for sc in result.results:
        assert sc.chunk is not None
        if "outlook" in sc.chunk.text.lower():
            ranked.append(sc.chunk.source_id)
    return ranked


def test_answer_chunk_loses_to_rivals_without_contextual_retrieval() -> None:
    ranked = _ranked_outlook_sources(contextual=False)
    assert set(ranked) == {"acme-annual", "globex", "initech"}
    # The real answer (acme-annual) — which never says "Acme" in its body — is the WORST of the
    # three outlook chunks: it only matches the common term "outlook".
    assert ranked[-1] == "acme-annual"


def test_contextual_retrieval_lifts_answer_chunk_above_rivals() -> None:
    ranked = _ranked_outlook_sources(contextual=True)
    assert set(ranked) == {"acme-annual", "globex", "initech"}
    # Contextual BM25 injects the rare "acme" term → the answer chunk now beats every rival.
    assert ranked[0] == "acme-annual"


def test_contextual_lift_is_genuine_not_a_constant() -> None:
    # Same corpus, same query, only the flag differs → the answer chunk's rank must flip from
    # worst to best among the outlook chunks.
    assert _ranked_outlook_sources(contextual=False)[-1] == "acme-annual"
    assert _ranked_outlook_sources(contextual=True)[0] == "acme-annual"


def test_default_off_preserves_original_chunk_text() -> None:
    # Provenance invariant: contextualization never mutates Chunk.text (citations quote text).
    settings = Settings(profile=Profile.TEST, use_contextual_retrieval=True)
    system = build_system(settings)
    system.ingest.ingest(
        SourceRef(
            source_id="acme-annual",
            media_type=MediaType.MARKDOWN,
            content=CORPUS["acme-annual"].encode(),
        )
    )
    result = system.query.retrieve(Query(text=QUERY, top_k=5, mode=RetrievalMode.SPARSE))
    target = next(sc for sc in result.results if sc.chunk and sc.chunk.source_id == "acme-annual")
    assert target.chunk is not None
    assert "Acme" not in target.chunk.text  # original text is untouched
    assert target.chunk.context is not None and "Acme" in target.chunk.context
    assert target.chunk.embedding_text.startswith(target.chunk.context)
