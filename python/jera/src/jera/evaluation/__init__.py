"""Evaluation harness: run datasets through the pipeline and score retrieval modes.

(The metric *contracts* and dataset shapes live in ``jera.evaluation_contracts``; this
package is the runnable machinery on top of them.)
"""

from jera.evaluation.ablation import (
    AblationCase,
    AblationConfig,
    AblationEntry,
    AblationReport,
    AblationRunner,
)
from jera.evaluation.computation import (
    ComputationCaseResult,
    ComputationEval,
    ComputationReport,
)
from jera.evaluation.dataset_builder import CaseSpec, build_gold_dataset
from jera.evaluation.generation_runner import GenerationEvalRunner
from jera.evaluation.gold_builder import ClaudeGoldGenerator, operand_in_chunk
from jera.evaluation.matrix import MatrixReport, StrategyEntry, run_matrix
from jera.evaluation.parser_bench import ParserBenchReport, ParserBenchResult, grade
from jera.evaluation.profiling import CorpusScenario, ProfileReport, run_profile
from jera.evaluation.report import (
    CaseResult,
    EvalReport,
    GenerationCaseResult,
    GenerationReport,
    ModeReport,
)
from jera.evaluation.runner import EvalRunner

__all__ = [
    "AblationCase",
    "AblationConfig",
    "AblationEntry",
    "AblationReport",
    "AblationRunner",
    "CaseResult",
    "CorpusScenario",
    "ProfileReport",
    "CaseSpec",
    "ClaudeGoldGenerator",
    "ComputationCaseResult",
    "ComputationEval",
    "ComputationReport",
    "EvalReport",
    "EvalRunner",
    "GenerationCaseResult",
    "GenerationEvalRunner",
    "GenerationReport",
    "MatrixReport",
    "ModeReport",
    "ParserBenchReport",
    "ParserBenchResult",
    "StrategyEntry",
    "build_gold_dataset",
    "grade",
    "operand_in_chunk",
    "run_matrix",
    "run_profile",
]
