"""Regression detection for evaluation runs.

Compares eval runs and detects performance degradations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from supabase import Client

from eval.models import EvalRunMetrics
from eval.supabase_client import EvalSupabaseClient


@dataclass
class RegressionAlert:
    """Represents a detected regression."""
    
    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    severity: str  # "warning", "critical"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "baseline": self.baseline_value,
            "current": self.current_value,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "severity": self.severity,
        }


@dataclass
class TaskRegression:
    """Regression for a specific task."""
    
    task_id: str
    baseline_score: float
    current_score: float
    baseline_status: str
    current_status: str
    
    @property
    def regressed(self) -> bool:
        return self.current_score < self.baseline_score or (
            self.baseline_status == "completed" and self.current_status != "completed"
        )


@dataclass
class RegressionReport:
    """Full regression comparison report."""
    
    baseline_run_id: UUID
    current_run_id: UUID
    baseline_suite: str
    current_suite: str
    generated_at: datetime = field(default_factory=datetime.now)
    
    # Overall metrics comparison
    success_rate_delta: float = 0.0
    average_score_delta: float = 0.0
    cost_delta: float = 0.0
    duration_delta: float = 0.0
    
    # Alerts (significant regressions)
    alerts: list[RegressionAlert] = field(default_factory=list)
    
    # Per-task regressions
    task_regressions: list[TaskRegression] = field(default_factory=list)
    
    # Summary
    total_tasks_compared: int = 0
    tasks_regressed: int = 0
    tasks_improved: int = 0
    tasks_unchanged: int = 0
    
    @property
    def has_regressions(self) -> bool:
        return len(self.alerts) > 0 or self.tasks_regressed > 0
    
    @property
    def has_critical_regressions(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": str(self.baseline_run_id),
            "current_run_id": str(self.current_run_id),
            "baseline_suite": self.baseline_suite,
            "current_suite": self.current_suite,
            "generated_at": self.generated_at.isoformat(),
            "metrics": {
                "success_rate_delta": self.success_rate_delta,
                "average_score_delta": self.average_score_delta,
                "cost_delta": self.cost_delta,
                "duration_delta": self.duration_delta,
            },
            "alerts": [a.to_dict() for a in self.alerts],
            "summary": {
                "total_tasks_compared": self.total_tasks_compared,
                "tasks_regressed": self.tasks_regressed,
                "tasks_improved": self.tasks_improved,
                "tasks_unchanged": self.tasks_unchanged,
                "has_regressions": self.has_regressions,
                "has_critical_regressions": self.has_critical_regressions,
            },
            "task_regressions": [
                {
                    "task_id": t.task_id,
                    "baseline_score": t.baseline_score,
                    "current_score": t.current_score,
                    "baseline_status": t.baseline_status,
                    "current_status": t.current_status,
                }
                for t in self.task_regressions
            ],
        }


class RegressionDetector:
    """Detects regressions between eval runs."""
    
    # Thresholds for alerting
    WARNING_THRESHOLD = 0.05  # 5% degradation
    CRITICAL_THRESHOLD = 0.15  # 15% degradation
    
    def __init__(
        self,
        supabase: Client,
        *,
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
    ):
        self._supabase = EvalSupabaseClient(supabase)
        self._warning_threshold = warning_threshold or self.WARNING_THRESHOLD
        self._critical_threshold = critical_threshold or self.CRITICAL_THRESHOLD
    
    def compare_runs(
        self,
        baseline_run_id: UUID,
        current_run_id: UUID,
    ) -> RegressionReport:
        """Compare two eval runs and detect regressions.
        
        Args:
            baseline_run_id: The baseline (previous) run to compare against
            current_run_id: The current (new) run to check for regressions
            
        Returns:
            RegressionReport with detailed comparison
        """
        baseline_run = self._supabase.get_eval_run(baseline_run_id)
        current_run = self._supabase.get_eval_run(current_run_id)
        
        if not baseline_run or not current_run:
            raise ValueError("One or both runs not found")
        
        report = RegressionReport(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            baseline_suite=baseline_run.name or "unknown",
            current_suite=current_run.name or "unknown",
        )
        
        # Compare overall metrics
        baseline_metrics = baseline_run.metrics or EvalRunMetrics()
        current_metrics = current_run.metrics or EvalRunMetrics()
        
        self._compare_metrics(report, baseline_metrics, current_metrics)
        
        # Compare individual tasks
        baseline_results = self._supabase.get_results_for_run(baseline_run_id)
        current_results = self._supabase.get_results_for_run(current_run_id)
        
        self._compare_tasks(report, baseline_results, current_results)
        
        return report
    
    def _compare_metrics(
        self,
        report: RegressionReport,
        baseline: EvalRunMetrics,
        current: EvalRunMetrics,
    ) -> None:
        """Compare aggregate metrics and generate alerts."""
        # Success rate comparison
        report.success_rate_delta = current.success_rate - baseline.success_rate
        if baseline.success_rate > 0:
            delta_pct = report.success_rate_delta / baseline.success_rate
            if delta_pct < -self._critical_threshold:
                report.alerts.append(RegressionAlert(
                    metric_name="success_rate",
                    baseline_value=baseline.success_rate,
                    current_value=current.success_rate,
                    delta=report.success_rate_delta,
                    delta_percent=delta_pct * 100,
                    severity="critical",
                ))
            elif delta_pct < -self._warning_threshold:
                report.alerts.append(RegressionAlert(
                    metric_name="success_rate",
                    baseline_value=baseline.success_rate,
                    current_value=current.success_rate,
                    delta=report.success_rate_delta,
                    delta_percent=delta_pct * 100,
                    severity="warning",
                ))
        
        # Average score comparison
        report.average_score_delta = current.average_score - baseline.average_score
        if baseline.average_score > 0:
            delta_pct = report.average_score_delta / baseline.average_score
            if delta_pct < -self._critical_threshold:
                report.alerts.append(RegressionAlert(
                    metric_name="average_score",
                    baseline_value=baseline.average_score,
                    current_value=current.average_score,
                    delta=report.average_score_delta,
                    delta_percent=delta_pct * 100,
                    severity="critical",
                ))
            elif delta_pct < -self._warning_threshold:
                report.alerts.append(RegressionAlert(
                    metric_name="average_score",
                    baseline_value=baseline.average_score,
                    current_value=current.average_score,
                    delta=report.average_score_delta,
                    delta_percent=delta_pct * 100,
                    severity="warning",
                ))
        
        # Cost comparison (increase is a regression)
        report.cost_delta = current.total_cost_usd - baseline.total_cost_usd
        if baseline.total_cost_usd > 0:
            delta_pct = report.cost_delta / baseline.total_cost_usd
            if delta_pct > self._critical_threshold * 2:  # More lenient for cost
                report.alerts.append(RegressionAlert(
                    metric_name="total_cost",
                    baseline_value=baseline.total_cost_usd,
                    current_value=current.total_cost_usd,
                    delta=report.cost_delta,
                    delta_percent=delta_pct * 100,
                    severity="warning",
                ))
        
        # Duration comparison (increase is a regression)
        report.duration_delta = (
            current.average_duration_seconds - baseline.average_duration_seconds
        )
    
    def _compare_tasks(
        self,
        report: RegressionReport,
        baseline_results: list,
        current_results: list,
    ) -> None:
        """Compare individual task results."""
        baseline_by_task = {r.eval_task_id: r for r in baseline_results}
        current_by_task = {r.eval_task_id: r for r in current_results}
        
        common_tasks = set(baseline_by_task.keys()) & set(current_by_task.keys())
        report.total_tasks_compared = len(common_tasks)
        
        for task_id in common_tasks:
            baseline = baseline_by_task[task_id]
            current = current_by_task[task_id]
            
            task_reg = TaskRegression(
                task_id=task_id,
                baseline_score=baseline.score or 0.0,
                current_score=current.score or 0.0,
                baseline_status=baseline.status,
                current_status=current.status,
            )
            
            if task_reg.regressed:
                report.tasks_regressed += 1
                report.task_regressions.append(task_reg)
            elif task_reg.current_score > task_reg.baseline_score:
                report.tasks_improved += 1
            else:
                report.tasks_unchanged += 1
    
    def get_latest_baseline(self, suite_name: str) -> UUID | None:
        """Get the most recent completed run for a suite to use as baseline.
        
        Args:
            suite_name: The eval suite name
            
        Returns:
            Run ID of the latest completed run, or None
        """
        runs = self._supabase._supabase.table("eval_runs").select(
            "id"
        ).eq("name", suite_name).eq("status", "completed").order(
            "finished_at", desc=True
        ).limit(1).execute()
        
        if runs.data:
            return UUID(runs.data[0]["id"])
        return None


def format_regression_report(report: RegressionReport) -> str:
    """Format a regression report as human-readable text."""
    lines = [
        "=" * 60,
        "REGRESSION REPORT",
        "=" * 60,
        f"Baseline: {report.baseline_suite} ({str(report.baseline_run_id)[:8]}...)",
        f"Current:  {report.current_suite} ({str(report.current_run_id)[:8]}...)",
        f"Generated: {report.generated_at.isoformat()}",
        "",
    ]
    
    # Summary
    lines.extend([
        "SUMMARY",
        "-" * 40,
        f"Tasks compared: {report.total_tasks_compared}",
        f"Regressed: {report.tasks_regressed}",
        f"Improved: {report.tasks_improved}",
        f"Unchanged: {report.tasks_unchanged}",
        "",
    ])
    
    # Metrics
    lines.extend([
        "METRICS DELTA",
        "-" * 40,
        f"Success Rate: {report.success_rate_delta:+.2%}",
        f"Average Score: {report.average_score_delta:+.3f}",
        f"Total Cost: ${report.cost_delta:+.4f}",
        f"Avg Duration: {report.duration_delta:+.1f}s",
        "",
    ])
    
    # Alerts
    if report.alerts:
        lines.extend([
            "ALERTS",
            "-" * 40,
        ])
        for alert in report.alerts:
            icon = "🔴" if alert.severity == "critical" else "🟡"
            lines.append(
                f"{icon} {alert.metric_name}: {alert.baseline_value:.3f} → "
                f"{alert.current_value:.3f} ({alert.delta_percent:+.1f}%)"
            )
        lines.append("")
    
    # Task regressions
    if report.task_regressions:
        lines.extend([
            "TASK REGRESSIONS",
            "-" * 40,
        ])
        for reg in report.task_regressions[:10]:  # Show top 10
            lines.append(
                f"  {reg.task_id}: {reg.baseline_score:.2f} → {reg.current_score:.2f} "
                f"({reg.baseline_status} → {reg.current_status})"
            )
        if len(report.task_regressions) > 10:
            lines.append(f"  ... and {len(report.task_regressions) - 10} more")
    
    # Final verdict
    lines.extend([
        "",
        "=" * 60,
    ])
    if report.has_critical_regressions:
        lines.append("❌ CRITICAL REGRESSIONS DETECTED")
    elif report.has_regressions:
        lines.append("⚠️  REGRESSIONS DETECTED")
    else:
        lines.append("✅ NO REGRESSIONS DETECTED")
    lines.append("=" * 60)
    
    return "\n".join(lines)


__all__ = [
    "RegressionAlert",
    "RegressionDetector",
    "RegressionReport",
    "TaskRegression",
    "format_regression_report",
]
