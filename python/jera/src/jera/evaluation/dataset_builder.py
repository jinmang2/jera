"""Deterministic gold-dataset builder.

Ingests a corpus, then labels gold chunks by substring containment — the chunk(s) whose
text contains a known answer marker become the relevance judgments. Fully reproducible, so
eval datasets are not hand-maintained id lists that rot when chunk ids change.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.config.registry import RagSystem
from jera.domain.document import MediaType, SourceRef
from jera.evaluation_contracts.dataset import EvalCase, EvalDataset, GoldChunk


class CaseSpec:
    """A query plus the answer substring that identifies its gold chunk(s)."""

    def __init__(self, case_id: str, query: str, answer_contains: str) -> None:
        self.case_id = case_id
        self.query = query
        self.answer_contains = answer_contains


def build_gold_dataset(
    system: RagSystem,
    *,
    name: str,
    documents: dict[str, tuple[MediaType, str]],
    cases: Sequence[CaseSpec],
) -> EvalDataset:
    """Ingest ``documents`` (source_id → (media_type, text)) and build gold labels for ``cases``."""
    for source_id, (media_type, text) in documents.items():
        system.ingest.ingest(
            SourceRef(source_id=source_id, media_type=media_type, content=text.encode())
        )

    eval_cases: list[EvalCase] = []
    for spec in cases:
        gold = [GoldChunk(chunk_id=cid) for cid in _chunks_containing(system, spec.answer_contains)]
        if not gold:
            raise ValueError(
                f"case {spec.case_id!r}: no chunk contains {spec.answer_contains!r}; "
                "the gold label is unsatisfiable"
            )
        eval_cases.append(EvalCase(case_id=spec.case_id, query=spec.query, gold=gold))
    return EvalDataset(name=name, cases=eval_cases)


def _chunks_containing(system: RagSystem, needle: str) -> list[str]:
    # Pull every stored chunk via a broad sparse pass over the needle's own terms, then filter
    # by exact substring so labels do not depend on retrieval quality.
    from jera.domain.retrieval import Query, RetrievalMode

    probe = system.query.retrieve(Query(text=needle, mode=RetrievalMode.SPARSE, top_k=1000))
    return [
        r.chunk_id
        for r in probe.results
        if r.chunk is not None and needle.lower() in r.chunk.text.lower()
    ]
