"""Evaluation contracts: dataset shapes and metric functions."""

from jera.evaluation_contracts.dataset import CaseKind, EvalCase, EvalDataset, GoldChunk
from jera.evaluation_contracts.metrics import (
    citation_faithfulness,
    mrr,
    ndcg_at_k,
    numeric_accuracy,
    recall_at_k,
)

__all__ = [
    "CaseKind",
    "EvalCase",
    "EvalDataset",
    "GoldChunk",
    "citation_faithfulness",
    "mrr",
    "ndcg_at_k",
    "numeric_accuracy",
    "recall_at_k",
]
