"""Evaluation harness for agent-cluster.

This module provides:
- scoring: ExactMatchScorer, TestSuiteScorer, RubricScorer
- regression: Regression detection between eval runs
- pricing: Model cost calculations
- controller: Evaluation orchestration
- swebench_loader: SWE-bench dataset loading
"""

from .models import (
    EvalQuestion,
    EvalResult,
    EvalRun,
    EvalRunSummary,
    EvalStatus,
    ScoringMode,
    CostTier,
)

__all__ = [
    "EvalQuestion",
    "EvalResult",
    "EvalRun",
    "EvalRunSummary",
    "EvalStatus",
    "ScoringMode",
    "CostTier",
]
