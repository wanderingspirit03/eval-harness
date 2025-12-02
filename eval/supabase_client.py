"""Supabase client for evaluation-related tables."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from supabase import Client

from eval.models import (
    CostTier,
    EvalQuestion,
    EvalResult,
    EvalRun,
    EvalRunMetrics,
    EvalStatus,
    ScoringMode,
)
from eval.observability import traced_span


class EvalSupabaseError(Exception):
    """Base exception for eval Supabase operations."""


class EvalSupabaseClient:
    """Thin wrapper around Supabase for eval-related tables.

    Handles operations on:
    - eval_questions: Source of eval questions and expected outputs
    - eval_runs: Metadata for each evaluation execution
    - eval_results: Per-task evaluation outcomes
    """

    def __init__(self, supabase: Client):
        self._client = supabase

    def get_question(self, question_id: str) -> EvalQuestion | None:
        """Fetch an eval question by ID.

        Args:
            question_id: The unique question identifier

        Returns:
            EvalQuestion if found, None otherwise
        """
        with traced_span("eval.supabase.get_question", input={"question_id": question_id}):
            response = (
                self._client.table("eval_questions")
                .select("*")
                .eq("id", question_id)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            if not rows:
                return None

            row = rows[0]
            return EvalQuestion(
                id=row["id"],
                prompt=row.get("prompt", ""),
                category=row.get("category", "general"),
                work_area=row.get("work_area", "general"),
                cost_tier=CostTier(row.get("cost_tier", "cheap")),
                expected_output=row.get("expected_output"),
                rubric=row.get("rubric"),
                scoring_mode=ScoringMode(row.get("scoring_mode", "exact_match")),
                metadata=row.get("metadata", {}),
            )

    def get_questions_by_ids(self, question_ids: list[str]) -> list[EvalQuestion]:
        """Fetch multiple eval questions by their IDs.

        Args:
            question_ids: List of question identifiers

        Returns:
            List of EvalQuestion objects (in no guaranteed order)
        """
        if not question_ids:
            return []

        with traced_span("eval.supabase.get_questions", input={"count": len(question_ids)}):
            response = (
                self._client.table("eval_questions")
                .select("*")
                .in_("id", question_ids)
                .execute()
            )
            rows = response.data or []

            return [
                EvalQuestion(
                    id=row["id"],
                    prompt=row.get("prompt", ""),
                    category=row.get("category", "general"),
                    work_area=row.get("work_area", "general"),
                    cost_tier=CostTier(row.get("cost_tier", "cheap")),
                    expected_output=row.get("expected_output"),
                    rubric=row.get("rubric"),
                    scoring_mode=ScoringMode(row.get("scoring_mode", "exact_match")),
                    metadata=row.get("metadata", {}),
                )
                for row in rows
            ]

    def get_questions_by_category(
        self,
        category: str,
        *,
        limit: int = 100,
    ) -> list[EvalQuestion]:
        """Fetch eval questions by category.

        Args:
            category: The category to filter by
            limit: Maximum number of questions to return

        Returns:
            List of EvalQuestion objects
        """
        with traced_span("eval.supabase.get_questions_by_category", input={"category": category}):
            response = (
                self._client.table("eval_questions")
                .select("*")
                .eq("category", category)
                .limit(limit)
                .execute()
            )
            rows = response.data or []

            return [
                EvalQuestion(
                    id=row["id"],
                    prompt=row.get("prompt", ""),
                    category=row.get("category", "general"),
                    work_area=row.get("work_area", "general"),
                    cost_tier=CostTier(row.get("cost_tier", "cheap")),
                    expected_output=row.get("expected_output"),
                    rubric=row.get("rubric"),
                    scoring_mode=ScoringMode(row.get("scoring_mode", "exact_match")),
                    metadata=row.get("metadata", {}),
                )
                for row in rows
            ]

    def create_eval_run(self, run: EvalRun) -> EvalRun:
        """Create a new eval run record.

        Args:
            run: EvalRun object with run configuration

        Returns:
            The created EvalRun with populated ID

        Raises:
            EvalSupabaseError: If the insert fails
        """
        with traced_span("eval.supabase.create_run", input={"name": run.name}):
            payload = {
                "id": str(run.id),
                "name": run.name,
                "version": run.version,
                "status": run.status.value,
                "total_tasks": run.total_tasks,
                "metrics": run.metrics.model_dump(),
                "created_at": run.created_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "metadata": run.metadata,
            }

            response = self._client.table("eval_runs").insert(payload).execute()
            if getattr(response, "error", None) is not None:
                raise EvalSupabaseError(f"Failed to create eval run: {response.error}")

            return run

    def update_eval_run(
        self,
        run_id: UUID,
        *,
        status: EvalStatus | None = None,
        metrics: EvalRunMetrics | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        """Update an existing eval run.

        Args:
            run_id: The run UUID
            status: New status (if provided)
            metrics: Updated metrics (if provided)
            finished_at: Completion timestamp (if provided)

        Raises:
            EvalSupabaseError: If the update fails
        """
        with traced_span("eval.supabase.update_run", input={"run_id": str(run_id)}):
            update_data: dict[str, Any] = {}

            if status is not None:
                update_data["status"] = status.value
            if metrics is not None:
                update_data["metrics"] = metrics.model_dump()
            if finished_at is not None:
                update_data["finished_at"] = finished_at.isoformat()

            if not update_data:
                return

            response = (
                self._client.table("eval_runs")
                .update(update_data)
                .eq("id", str(run_id))
                .execute()
            )
            if getattr(response, "error", None) is not None:
                raise EvalSupabaseError(f"Failed to update eval run: {response.error}")

    def get_eval_run(self, run_id: UUID) -> EvalRun | None:
        """Fetch an eval run by ID.

        Args:
            run_id: The run UUID

        Returns:
            EvalRun if found, None otherwise
        """
        with traced_span("eval.supabase.get_run", input={"run_id": str(run_id)}):
            response = (
                self._client.table("eval_runs")
                .select("*")
                .eq("id", str(run_id))
                .limit(1)
                .execute()
            )
            rows = response.data or []
            if not rows:
                return None

            row = rows[0]
            return EvalRun(
                id=UUID(row["id"]),
                name=row["name"],
                version=row.get("version", "1.0.0"),
                status=EvalStatus(row.get("status", "pending")),
                total_tasks=row.get("total_tasks", 0),
                metrics=EvalRunMetrics(**row.get("metrics", {})),
                created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                finished_at=(
                    datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00"))
                    if row.get("finished_at")
                    else None
                ),
                metadata=row.get("metadata", {}),
            )

    def create_eval_result(
        self,
        result: EvalResult,
        *,
        agent_output: str | None = None,
    ) -> EvalResult:
        """Create a new eval result record.

        Args:
            result: EvalResult object
            agent_output: The agent's raw output (stored in dedicated column)

        Returns:
            The created EvalResult

        Raises:
            EvalSupabaseError: If the insert fails
        """
        with traced_span(
            "eval.supabase.create_result",
            input={"eval_task_id": result.eval_task_id},
        ):
            payload = {
                "id": str(result.id),
                "eval_run_id": str(result.eval_run_id),
                "eval_task_id": result.eval_task_id,
                "gateway_task_id": str(result.gateway_task_id) if result.gateway_task_id else None,
                "work_area": result.work_area,
                "cost_tier": result.cost_tier,
                "status": result.status,
                "score": result.score,
                "raw_score": result.raw_score,
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "finished_at": result.finished_at.isoformat() if result.finished_at else None,
                "duration_seconds": result.duration_seconds,
                "cost": result.cost,
                "traces": result.traces,
                "errors": result.errors,
                "reproduction": result.reproduction,
                "metadata": result.metadata,
                "agent_output": agent_output,  # Dedicated column for benchmarking
            }

            response = self._client.table("eval_results").insert(payload).execute()
            if getattr(response, "error", None) is not None:
                raise EvalSupabaseError(f"Failed to create eval result: {response.error}")

            return result

    def get_results_for_run(self, run_id: UUID) -> list[EvalResult]:
        """Fetch all eval results for a run.

        Args:
            run_id: The run UUID

        Returns:
            List of EvalResult objects
        """
        with traced_span("eval.supabase.get_results", input={"run_id": str(run_id)}):
            response = (
                self._client.table("eval_results")
                .select("*")
                .eq("eval_run_id", str(run_id))
                .execute()
            )
            rows = response.data or []

            results = []
            for row in rows:
                results.append(
                    EvalResult(
                        id=UUID(row["id"]),
                        eval_run_id=UUID(row["eval_run_id"]),
                        eval_task_id=row["eval_task_id"],
                        gateway_task_id=UUID(row["gateway_task_id"]) if row.get("gateway_task_id") else None,
                        work_area=row.get("work_area", "general"),
                        cost_tier=row.get("cost_tier", "cheap"),
                        status=row.get("status", "pending"),
                        score=row.get("score", 0.0),
                        raw_score=row.get("raw_score", {}),
                        started_at=(
                            datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
                            if row.get("started_at")
                            else None
                        ),
                        finished_at=(
                            datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00"))
                            if row.get("finished_at")
                            else None
                        ),
                        duration_seconds=row.get("duration_seconds"),
                        cost=row.get("cost", {}),
                        traces=row.get("traces", {}),
                        errors=row.get("errors", {}),
                        reproduction=row.get("reproduction", {}),
                        metadata=row.get("metadata", {}),
                    )
                )

            return results

    def aggregate_run_metrics(
        self,
        run_id: UUID,
        success_threshold: float = 0.8,
    ) -> EvalRunMetrics:
        """Compute aggregated metrics for an eval run.

        Args:
            run_id: The run UUID
            success_threshold: Score threshold for success classification

        Returns:
            EvalRunMetrics with aggregated statistics
        """
        with traced_span("eval.supabase.aggregate_metrics", input={"run_id": str(run_id)}):
            results = self.get_results_for_run(run_id)

            if not results:
                return EvalRunMetrics()

            total = len(results)
            completed = sum(1 for r in results if r.status == "completed")
            failed = sum(1 for r in results if r.status == "failed")
            timed_out = sum(1 for r in results if r.status == "timed_out")

            scores = [r.score for r in results if r.status == "completed"]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            success_rate = sum(1 for s in scores if s >= success_threshold) / len(scores) if scores else 0.0

            durations = [r.duration_seconds for r in results if r.duration_seconds is not None]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            total_cost = sum(r.cost.get("usd", 0.0) for r in results)

            by_work_area: dict[str, dict[str, float]] = {}
            by_cost_tier: dict[str, dict[str, float]] = {}

            for area in set(r.work_area for r in results):
                area_results = [r for r in results if r.work_area == area and r.status == "completed"]
                if area_results:
                    area_scores = [r.score for r in area_results]
                    by_work_area[area] = {
                        "average_score": sum(area_scores) / len(area_scores),
                        "success_rate": sum(1 for s in area_scores if s >= success_threshold) / len(area_scores),
                        "count": len(area_results),
                    }

            for tier in set(r.cost_tier for r in results):
                tier_results = [r for r in results if r.cost_tier == tier and r.status == "completed"]
                if tier_results:
                    tier_scores = [r.score for r in tier_results]
                    tier_costs = [r.cost.get("usd", 0.0) for r in tier_results]
                    by_cost_tier[tier] = {
                        "average_score": sum(tier_scores) / len(tier_scores),
                        "average_cost_usd": sum(tier_costs) / len(tier_costs) if tier_costs else 0.0,
                        "count": len(tier_results),
                    }

            return EvalRunMetrics(
                total_tasks=total,
                completed_tasks=completed,
                failed_tasks=failed,
                timed_out_tasks=timed_out,
                average_score=avg_score,
                success_rate=success_rate,
                total_cost_usd=total_cost,
                average_duration_seconds=avg_duration,
                by_work_area=by_work_area,
                by_cost_tier=by_cost_tier,
            )


__all__ = [
    "EvalSupabaseClient",
    "EvalSupabaseError",
]
