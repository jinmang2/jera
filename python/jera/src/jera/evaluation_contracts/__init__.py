"""Evaluation contracts: dataset shapes and metric functions."""

from jera.evaluation_contracts.dataset import CaseKind, EvalCase, EvalDataset, GoldChunk
from jera.evaluation_contracts.generation_metrics import (
    answer_correctness,
    answer_relevance,
    context_precision,
    faithfulness,
)
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
    "answer_correctness",
    "answer_relevance",
    "citation_faithfulness",
    "context_precision",
    "faithfulness",
    "mrr",
    "ndcg_at_k",
    "numeric_accuracy",
    "recall_at_k",
]
