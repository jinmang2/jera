"""Evaluation harness: run datasets through the pipeline and score retrieval modes.

(The metric *contracts* and dataset shapes live in ``jera.evaluation_contracts``; this
package is the runnable machinery on top of them.)
"""

from jera.evaluation.dataset_builder import CaseSpec, build_gold_dataset
from jera.evaluation.report import CaseResult, EvalReport, ModeReport
from jera.evaluation.runner import EvalRunner

__all__ = [
    "CaseResult",
    "CaseSpec",
    "EvalReport",
    "EvalRunner",
    "ModeReport",
    "build_gold_dataset",
]
