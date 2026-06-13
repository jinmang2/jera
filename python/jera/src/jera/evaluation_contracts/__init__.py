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
from jera.evaluation_contracts.ragchecker_metrics import (
    abstention_score,
    citation_precision,
    citation_recall,
    claim_precision,
    claim_recall,
    noise_sensitivity,
)

__all__ = [
    "CaseKind",
    "EvalCase",
    "EvalDataset",
    "GoldChunk",
    "abstention_score",
    "answer_correctness",
    "answer_relevance",
    "citation_faithfulness",
    "citation_precision",
    "citation_recall",
    "claim_precision",
    "claim_recall",
    "context_precision",
    "faithfulness",
    "mrr",
    "ndcg_at_k",
    "noise_sensitivity",
    "numeric_accuracy",
    "recall_at_k",
]
