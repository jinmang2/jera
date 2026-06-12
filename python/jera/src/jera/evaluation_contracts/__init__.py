"""Evaluation contracts: dataset shapes and metric functions."""

from jera.evaluation_contracts.dataset import EvalCase, EvalDataset, GoldChunk
from jera.evaluation_contracts.metrics import (
    citation_faithfulness,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "GoldChunk",
    "citation_faithfulness",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
