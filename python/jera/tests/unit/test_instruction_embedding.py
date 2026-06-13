"""InstructionEmbedding — the instruction STEERS retrieval (non-tautological).

The same neutral query text, wrapped with two different task instructions, must embed closer to
the chunk whose vocabulary the instruction shares. With the deterministic hash embedding this is a
real bag-of-tokens consequence (the instruction tokens mix into the query vector), not a mock.
"""

from __future__ import annotations

import math

from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.embedding.instruction import InstructionEmbedding
from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider

_CHUNK_A = "alpha bravo charlie delta"
_CHUNK_B = "echo foxtrot golf hotel"
_QUERY = "india"  # neutral: appears in neither chunk
_TASK_A = "retrieve documents about alpha bravo"
_TASK_B = "retrieve documents about echo foxtrot"


def _cos(a: DenseVector, b: DenseVector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _base() -> HashEmbedding:
    return HashEmbedding(dimensions=256)


def test_instruction_steers_query_toward_matching_chunk() -> None:
    base = _base()
    # Documents embedded as-is (instruction is query-side only).
    vec_a, vec_b = base.embed([_CHUNK_A, _CHUNK_B])

    q_with_a = InstructionEmbedding(base, task=_TASK_A).embed_query(_QUERY)
    q_with_b = InstructionEmbedding(base, task=_TASK_B).embed_query(_QUERY)

    # Instruction A (mentions alpha/bravo) pulls the same query toward chunk A; B toward chunk B.
    assert _cos(q_with_a, vec_a) > _cos(q_with_a, vec_b)
    assert _cos(q_with_b, vec_b) > _cos(q_with_b, vec_a)


def test_documents_are_embedded_without_instruction_prefix() -> None:
    base = _base()
    wrapped = InstructionEmbedding(base, task=_TASK_A)
    assert wrapped.embed([_CHUNK_A]) == base.embed([_CHUNK_A])  # docs unchanged


def test_is_an_embedding_provider_with_delegated_metadata() -> None:
    base = _base()
    wrapped = InstructionEmbedding(base, task=_TASK_A)
    assert isinstance(wrapped, EmbeddingProvider)
    assert wrapped.dimensions == base.dimensions
    assert wrapped.context_limit == base.context_limit
    assert wrapped.model_id == f"{base.model_id}-instruct"


def test_query_format_matches_qwen3_e5_convention() -> None:
    wrapped = InstructionEmbedding(_base(), task="my task")
    assert wrapped._format_query("hello") == "Instruct: my task\nQuery: hello"
