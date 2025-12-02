"""Evaluation controller for orchestrating eval runs."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import UUID

from supabase import Client

from eval.client import (
    GatewayClient,
    GatewayClientError,
    GatewayTimeoutError,
)
from eval.models import (
    CostTier,
    EvalQuestion,
    EvalResult,
    EvalRun,
    EvalRunMetrics,
    EvalRunSummary,
    EvalStatus,
    EvalSuiteConfig,
    EvalTaskCost,
    EvalTaskResult,
)
from eval.scoring import ScoringEngine, ScoreResult
from eval.supabase_client import EvalSupabaseClient
from eval.observability import traced_span


class EvaluationControllerError(Exception):
    """Base exception for evaluation controller errors."""


class EvaluationController:
    """Orchestrates evaluation runs against the Task Gateway.

    Handles batch submission with concurrency control, result collection,
    scoring, and persistence to Supabase.
    """

    def __init__(
        self,
        supabase: Client,
        *,
        gateway_base_url: str | None = None,
        gateway_timeout_seconds: float = 30.0,
    ):
        self._supabase = EvalSupabaseClient(supabase)
        self._scoring_engine = ScoringEngine()
        self._gateway_base_url = gateway_base_url
        self._gateway_timeout = gateway_timeout_seconds

    async def run_eval_suite(
        self,
        suite_config: EvalSuiteConfig,
        *,
        max_concurrency: int | None = None,
    ) -> EvalRunSummary:
        """Run a complete evaluation suite.

        Args:
            suite_config: Configuration specifying which questions to run
            max_concurrency: Override suite's max_concurrency if provided

        Returns:
            EvalRunSummary with run metadata and all results
        """
        concurrency = max_concurrency or suite_config.max_concurrency
        loop = asyncio.get_running_loop()

        with traced_span(
            "eval.controller.run_suite",
            input={"suite": suite_config.name, "concurrency": concurrency},
        ):
            questions = await loop.run_in_executor(
                None,
                self._supabase.get_questions_by_ids,
                suite_config.question_ids,
            )
            if not questions:
                raise EvaluationControllerError(
                    f"No questions found for suite {suite_config.name}"
                )

            question_map = {q.id: q for q in questions}
            missing = set(suite_config.question_ids) - set(question_map.keys())
            if missing:
                pass

            run = EvalRun(
                name=suite_config.name,
                version=suite_config.version,
                status=EvalStatus.RUNNING,
                total_tasks=len(questions),
                metadata={
                    "environment": os.environ.get("APP_ENV", "local"),
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    **suite_config.metadata,
                },
            )

            try:
                await loop.run_in_executor(None, self._supabase.create_eval_run, run)
            except Exception:
                pass

            semaphore = asyncio.Semaphore(concurrency)
            results: list[EvalTaskResult] = []

            async def process_question(question: EvalQuestion) -> EvalTaskResult:
                async with semaphore:
                    return await self._run_single_eval(
                        run_id=run.id,
                        question=question,
                        suite_config=suite_config,
                    )

            try:
                tasks = [process_question(q) for q in questions]
                results = await asyncio.gather(*tasks, return_exceptions=False)

                metrics = self._compute_metrics(results, suite_config.success_threshold)
                run.metrics = metrics
                run.status = EvalStatus.COMPLETED
                run.finished_at = datetime.now(timezone.utc)

                try:
                    await loop.run_in_executor(
                        None,
                        partial(
                            self._supabase.update_eval_run,
                            run.id,
                            status=run.status,
                            metrics=metrics,
                            finished_at=run.finished_at,
                        ),
                    )
                except Exception:
                    pass

                passed = metrics.success_rate >= suite_config.success_threshold

                return EvalRunSummary(
                    run=run,
                    results=results,
                    passed=passed,
                )
            except Exception as exc:
                # Mark run as failed if an unexpected error occurs
                run.status = EvalStatus.FAILED
                run.finished_at = datetime.now(timezone.utc)
                run.metrics = EvalRunMetrics(
                    total_tasks=len(questions),
                    completed_tasks=len([r for r in results if r.status == EvalStatus.COMPLETED]),
                    failed_tasks=len([r for r in results if r.status == EvalStatus.FAILED]),
                )
                try:
                    await loop.run_in_executor(
                        None,
                        partial(
                            self._supabase.update_eval_run,
                            run.id,
                            status=run.status,
                            metrics=run.metrics,
                            finished_at=run.finished_at,
                        ),
                    )
                except Exception:
                    pass
                raise EvaluationControllerError(f"Eval run failed: {exc}") from exc

    async def _run_single_eval(
        self,
        run_id: UUID,
        question: EvalQuestion,
        suite_config: EvalSuiteConfig,
    ) -> EvalTaskResult:
        """Run a single evaluation task.

        Args:
            run_id: Parent eval run ID
            question: The eval question to execute
            suite_config: Suite configuration for timeouts etc.

        Returns:
            EvalTaskResult with score and metadata
        """
        with traced_span(
            "eval.controller.run_task",
            input={"question_id": question.id},
        ):
            result = EvalTaskResult(
                eval_task_id=question.id,
                work_area=question.work_area,
                cost_tier=question.cost_tier,
                started_at=datetime.now(timezone.utc),
            )

            task_data: dict[str, Any] = {}

            try:
                async with GatewayClient(
                    base_url=self._gateway_base_url,
                    timeout_seconds=self._gateway_timeout,
                    task_timeout_seconds=suite_config.task_timeout_seconds,
                    poll_interval_seconds=suite_config.poll_interval_seconds,
                ) as client:
                    task_data = await client.submit_and_wait(
                        title=f"eval:{question.id}",
                        description=question.prompt,
                        source="eval",
                        metadata={
                            "eval_suite": suite_config.name,
                            "eval_task_id": question.id,
                            "work_area": question.work_area,
                            "cost_tier": question.cost_tier.value,
                        },
                        timeout_seconds=suite_config.task_timeout_seconds,
                    )

                    result.gateway_task_id = UUID(task_data["id"])
                    gateway_status = task_data.get("status", "")

                    if gateway_status == "completed":
                        result.status = EvalStatus.COMPLETED
                    elif gateway_status == "failed":
                        result.status = EvalStatus.FAILED
                        result.errors = {
                            "gateway_error": task_data.get("error", "Unknown error"),
                        }
                    else:
                        result.status = EvalStatus.FAILED
                        result.errors = {"unexpected_status": gateway_status}

            except GatewayTimeoutError as exc:
                result.status = EvalStatus.TIMED_OUT
                result.errors = {"timeout": str(exc)}
            except GatewayClientError as exc:
                result.status = EvalStatus.FAILED
                result.errors = {"client_error": str(exc)}
            except Exception as exc:
                result.status = EvalStatus.FAILED
                result.errors = {"unexpected_error": str(exc)}

            result.finished_at = datetime.now(timezone.utc)
            if result.started_at and result.finished_at:
                result.duration_seconds = (
                    result.finished_at - result.started_at
                ).total_seconds()

            if result.status == EvalStatus.COMPLETED and task_data:
                try:
                    score_result = await self._scoring_engine.score(question, task_data)
                    result.score = score_result.score
                    result.raw_score = score_result.to_dict()
                except Exception as exc:
                    result.raw_score = {"scoring_error": str(exc)}

                result.cost = self._extract_cost(task_data)

            # Extract agent output for storage (enables benchmarking comparison)
            agent_output = None
            if task_data:
                agent_output = task_data.get("result_metadata", {}).get("answer_text")
                if agent_output is None:
                    agent_output = task_data.get("result")

            result.reproduction = {
                "prompt": question.prompt,
                "scoring_mode": question.scoring_mode.value,
                "expected_output": question.expected_output,
                "suite_config": {
                    "name": suite_config.name,
                    "version": suite_config.version,
                },
            }

            try:
                db_result = EvalResult(
                    eval_run_id=run_id,
                    eval_task_id=result.eval_task_id,
                    gateway_task_id=result.gateway_task_id,
                    work_area=result.work_area,
                    cost_tier=result.cost_tier.value,
                    status=result.status.value,
                    score=result.score,
                    raw_score=result.raw_score,
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                    duration_seconds=result.duration_seconds,
                    cost=result.cost.model_dump(),
                    traces=result.traces,
                    errors=result.errors,
                    reproduction=result.reproduction,
                    metadata=result.metadata,
                )
                # Pass agent_output separately for dedicated column storage
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    partial(
                        self._supabase.create_eval_result,
                        db_result,
                        agent_output=agent_output,
                    ),
                )
            except Exception:
                pass

            return result

    def _extract_cost(self, task_data: dict[str, Any]) -> EvalTaskCost:
        """Extract cost information from task result metadata."""
        metadata = task_data.get("result_metadata", {})
        cost_data = metadata.get("cost", {})

        return EvalTaskCost(
            usd=cost_data.get("usd", 0.0),
            tokens={
                "prompt": cost_data.get("tokens", {}).get("prompt", 0),
                "completion": cost_data.get("tokens", {}).get("completion", 0),
            },
        )

    def _compute_metrics(
        self,
        results: list[EvalTaskResult],
        success_threshold: float,
    ) -> EvalRunMetrics:
        """Compute aggregated metrics from results."""
        if not results:
            return EvalRunMetrics()

        total = len(results)
        completed = sum(1 for r in results if r.status == EvalStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == EvalStatus.FAILED)
        timed_out = sum(1 for r in results if r.status == EvalStatus.TIMED_OUT)

        completed_results = [r for r in results if r.status == EvalStatus.COMPLETED]
        scores = [r.score for r in completed_results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        success_rate = (
            sum(1 for s in scores if s >= success_threshold) / len(scores)
            if scores
            else 0.0
        )

        durations = [r.duration_seconds for r in results if r.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        total_cost = sum(r.cost.usd for r in results)

        by_work_area: dict[str, dict[str, float]] = {}
        for area in set(r.work_area for r in results):
            area_results = [
                r for r in completed_results if r.work_area == area
            ]
            if area_results:
                area_scores = [r.score for r in area_results]
                by_work_area[area] = {
                    "average_score": sum(area_scores) / len(area_scores),
                    "success_rate": (
                        sum(1 for s in area_scores if s >= success_threshold)
                        / len(area_scores)
                    ),
                    "count": float(len(area_results)),
                }

        by_cost_tier: dict[str, dict[str, float]] = {}
        for tier in set(r.cost_tier.value for r in results):
            tier_results = [
                r for r in completed_results if r.cost_tier.value == tier
            ]
            if tier_results:
                tier_scores = [r.score for r in tier_results]
                tier_costs = [r.cost.usd for r in tier_results]
                by_cost_tier[tier] = {
                    "average_score": sum(tier_scores) / len(tier_scores),
                    "average_cost_usd": (
                        sum(tier_costs) / len(tier_costs) if tier_costs else 0.0
                    ),
                    "count": float(len(tier_results)),
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

    async def get_run_summary(self, run_id: UUID) -> EvalRunSummary | None:
        """Retrieve a summary of a completed eval run.

        Args:
            run_id: The eval run ID

        Returns:
            EvalRunSummary if found, None otherwise
        """
        loop = asyncio.get_running_loop()
        run = await loop.run_in_executor(None, self._supabase.get_eval_run, run_id)
        if run is None:
            return None

        db_results = await loop.run_in_executor(
            None, self._supabase.get_results_for_run, run_id
        )

        results = [
            EvalTaskResult(
                eval_task_id=r.eval_task_id,
                gateway_task_id=r.gateway_task_id,
                work_area=r.work_area,
                cost_tier=CostTier(r.cost_tier),
                status=EvalStatus(r.status),
                score=r.score,
                raw_score=r.raw_score,
                started_at=r.started_at,
                finished_at=r.finished_at,
                duration_seconds=r.duration_seconds,
                cost=EvalTaskCost(**r.cost) if r.cost else EvalTaskCost(),
                traces=r.traces,
                errors=r.errors,
                reproduction=r.reproduction,
                metadata=r.metadata,
            )
            for r in db_results
        ]

        return EvalRunSummary(
            run=run,
            results=results,
            passed=run.metrics.success_rate >= 0.8,
        )


__all__ = [
    "EvaluationController",
    "EvaluationControllerError",
]
