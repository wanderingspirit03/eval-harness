"""Pydantic models for evaluation configs and runtime data."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ScoringMode(str, Enum):
    """Scoring mode for evaluation questions."""

    EXACT_MATCH = "exact_match"
    TEST_SUITE = "test_suite"
    RUBRIC = "rubric"


class CostTier(str, Enum):
    """Cost tier classification for evaluation tasks."""

    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"


class EvalStatus(str, Enum):
    """Status of an evaluation run or task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class EvalQuestion(BaseModel):
    """An evaluation question from the Eval Questions database."""

    id: str = Field(..., description="Unique eval question identifier")
    prompt: str = Field(..., description="Task prompt or problem description")
    category: str = Field(default="general", description="Category (e.g., eng, research)")
    work_area: str = Field(default="general", description="Work area classification")
    cost_tier: CostTier = Field(default=CostTier.CHEAP, description="Cost tier")
    expected_output: Optional[Any] = Field(default=None, description="Expected output for exact/test scoring")
    rubric: Optional[dict[str, Any]] = Field(default=None, description="Rubric for LLM-as-judge scoring")
    scoring_mode: ScoringMode = Field(default=ScoringMode.EXACT_MATCH, description="Scoring mode")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalSuiteConfig(BaseModel):
    """Configuration for an evaluation suite."""

    name: str = Field(..., description="Suite name (e.g., eng_smoke)")
    version: str = Field(default="1.0.0", description="Version string")
    question_ids: list[str] = Field(default_factory=list, description="Question IDs to include")
    max_concurrency: int = Field(default=5, ge=1, le=50, description="Max concurrent tasks")
    task_timeout_seconds: float = Field(default=300.0, ge=10.0, description="Per-task timeout")
    poll_interval_seconds: float = Field(default=2.0, ge=0.5, description="Polling interval")
    success_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Success threshold for pass/fail")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalTaskCost(BaseModel):
    """Cost information for an evaluation task."""

    usd: float = Field(default=0.0, ge=0.0, description="Cost in USD")
    tokens: dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0},
        description="Token counts",
    )


class EvalTaskResult(BaseModel):
    """Result of a single evaluation task."""

    eval_task_id: str = Field(..., description="Eval question ID")
    gateway_task_id: Optional[UUID] = Field(default=None, description="Gateway task ID")
    work_area: str = Field(default="general", description="Work area")
    cost_tier: CostTier = Field(default=CostTier.CHEAP, description="Cost tier")
    status: EvalStatus = Field(default=EvalStatus.PENDING, description="Task status")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized score (0-1)")
    raw_score: dict[str, Any] = Field(default_factory=dict, description="Raw scoring details")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    finished_at: Optional[datetime] = Field(default=None, description="End timestamp")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Duration in seconds")
    cost: EvalTaskCost = Field(default_factory=EvalTaskCost, description="Cost information")
    traces: dict[str, Any] = Field(default_factory=dict, description="Laminar trace IDs")
    errors: dict[str, Any] = Field(default_factory=dict, description="Error information")
    reproduction: dict[str, Any] = Field(default_factory=dict, description="Reproduction data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalRunMetrics(BaseModel):
    """Aggregated metrics for an evaluation run."""

    total_tasks: int = Field(default=0, ge=0, description="Total number of tasks")
    completed_tasks: int = Field(default=0, ge=0, description="Number of completed tasks")
    failed_tasks: int = Field(default=0, ge=0, description="Number of failed tasks")
    timed_out_tasks: int = Field(default=0, ge=0, description="Number of timed out tasks")
    average_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average score")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    average_duration_seconds: float = Field(default=0.0, ge=0.0, description="Average duration")
    by_work_area: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics grouped by work area",
    )
    by_cost_tier: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics grouped by cost tier",
    )


class EvalRun(BaseModel):
    """Metadata for an evaluation run."""

    id: UUID = Field(default_factory=uuid4, description="Run ID")
    name: str = Field(..., description="Suite name")
    version: str = Field(default="1.0.0", description="Version string")
    status: EvalStatus = Field(default=EvalStatus.PENDING, description="Run status")
    total_tasks: int = Field(default=0, ge=0, description="Expected number of tasks")
    metrics: EvalRunMetrics = Field(default_factory=EvalRunMetrics, description="Aggregated metrics")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation time")
    finished_at: Optional[datetime] = Field(default=None, description="Completion time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalResult(BaseModel):
    """Per-task eval result for storage in Supabase."""

    id: UUID = Field(default_factory=uuid4, description="Result ID")
    eval_run_id: UUID = Field(..., description="FK to eval_runs")
    eval_task_id: str = Field(..., description="Eval question ID")
    gateway_task_id: Optional[UUID] = Field(default=None, description="FK to tasks table")
    work_area: str = Field(default="general", description="Work area")
    cost_tier: str = Field(default="cheap", description="Cost tier")
    status: str = Field(default="pending", description="Task status")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized score")
    raw_score: dict[str, Any] = Field(default_factory=dict, description="Raw scoring details")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    finished_at: Optional[datetime] = Field(default=None, description="End timestamp")
    duration_seconds: Optional[float] = Field(default=None, description="Duration")
    cost: dict[str, Any] = Field(default_factory=dict, description="Cost JSON")
    traces: dict[str, Any] = Field(default_factory=dict, description="Trace IDs")
    errors: dict[str, Any] = Field(default_factory=dict, description="Errors")
    reproduction: dict[str, Any] = Field(default_factory=dict, description="Reproduction data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalRunSummary(BaseModel):
    """Summary returned after completing an evaluation run."""

    run: EvalRun = Field(..., description="Evaluation run metadata")
    results: list[EvalTaskResult] = Field(default_factory=list, description="Per-task results")
    passed: bool = Field(default=False, description="Whether the run passed success threshold")


__all__ = [
    "CostTier",
    "EvalQuestion",
    "EvalResult",
    "EvalRun",
    "EvalRunMetrics",
    "EvalRunSummary",
    "EvalStatus",
    "EvalSuiteConfig",
    "EvalTaskCost",
    "EvalTaskResult",
    "ScoringMode",
]
