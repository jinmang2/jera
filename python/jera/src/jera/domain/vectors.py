"""Vector value objects shared by providers and stores."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

DenseVector = list[float]


class SparseVector(BaseModel):
    """A sparse vector as parallel ``indices``/``values`` arrays.

    Mirrors Qdrant's sparse-vector shape so the in-memory and Qdrant adapters share a
    representation. Indices need not be sorted; equal-length is enforced.
    """

    model_config = {"frozen": True}

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def _check_lengths(self) -> SparseVector:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"sparse vector indices/values length mismatch: "
                f"{len(self.indices)} != {len(self.values)}"
            )
        return self

    def dot(self, other: SparseVector) -> float:
        """Sparse dot product over shared indices."""
        a = dict(zip(self.indices, self.values, strict=True))
        score = 0.0
        for idx, val in zip(other.indices, other.values, strict=True):
            hit = a.get(idx)
            if hit is not None:
                score += hit * val
        return score

    def is_empty(self) -> bool:
        return not self.indices
