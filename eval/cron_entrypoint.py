#!/usr/bin/env python3
"""Cron/CI-friendly entrypoint for scheduled evaluation runs.

Always outputs JSON for easy parsing by CI systems.
Designed to be run by cron jobs or CI/CD pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.database import get_supabase_client
from eval.env import load_project_env
from eval.controller import EvaluationController
from eval.models import EvalRunSummary, EvalSuiteConfig
from eval.regression import RegressionDetector

load_project_env()


EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_EVAL_FAILED = 2
EXIT_RUNTIME_ERROR = 3


def load_suite_config(config_path: str) -> EvalSuiteConfig:
    """Load suite configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    content = path.read_text()
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    elif suffix == ".json":
        data = json.loads(content)
    else:
        try:
            data = yaml.safe_load(content)
        except Exception:
            data = json.loads(content)

    return EvalSuiteConfig(**data)


def format_json_output(
    summary: EvalRunSummary | None,
    error: str | None = None,
    regression_detected: bool = False,
) -> dict[str, Any]:
    """Format output as a structured dict for JSON serialization."""
    output: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": summary is not None and summary.passed and not regression_detected,
    }

    if error:
        output["error"] = error

    if summary:
        output["run"] = {
            "id": str(summary.run.id),
            "name": summary.run.name,
            "version": summary.run.version,
            "status": summary.run.status.value,
            "total_tasks": summary.run.total_tasks,
            "created_at": summary.run.created_at.isoformat(),
            "finished_at": (
                summary.run.finished_at.isoformat() if summary.run.finished_at else None
            ),
        }
        output["metrics"] = {
            "total_tasks": summary.run.metrics.total_tasks,
            "completed_tasks": summary.run.metrics.completed_tasks,
            "failed_tasks": summary.run.metrics.failed_tasks,
            "timed_out_tasks": summary.run.metrics.timed_out_tasks,
            "average_score": summary.run.metrics.average_score,
            "success_rate": summary.run.metrics.success_rate,
            "total_cost_usd": summary.run.metrics.total_cost_usd,
            "average_duration_seconds": summary.run.metrics.average_duration_seconds,
        }
        output["passed"] = summary.passed
        output["regression_detected"] = regression_detected
        output["results"] = [
            {
                "eval_task_id": r.eval_task_id,
                "gateway_task_id": str(r.gateway_task_id) if r.gateway_task_id else None,
                "status": r.status.value,
                "score": r.score,
                "duration_seconds": r.duration_seconds,
                "errors": r.errors if r.errors else None,
            }
            for r in summary.results
        ]

    return output


async def run_cron_evaluation(args: argparse.Namespace) -> int:
    """Execute an evaluation suite and write JSON output."""
    summary: EvalRunSummary | None = None
    error: str | None = None
    regression_detected = False
    exit_code = EXIT_SUCCESS

    try:
        config = load_suite_config(args.config)
    except Exception as exc:
        error = f"Config error: {exc}"
        exit_code = EXIT_CONFIG_ERROR
        output = format_json_output(None, error=error)
        write_output(output, args.output)
        return exit_code

    try:
        supabase = get_supabase_client()
        controller = EvaluationController(
            supabase,
            gateway_base_url=args.gateway_url or os.environ.get("GATEWAY_BASE_URL"),
        )

        summary = await controller.run_eval_suite(config)

        if not summary.passed:
            exit_code = EXIT_EVAL_FAILED

        # Check for regressions if enabled
        if args.check_regression:
            detector = RegressionDetector(supabase)
            baseline_id = detector.get_latest_baseline(config.name)
            if baseline_id:
                report = detector.compare_runs(baseline_id, summary.run.id)
                if report.has_critical_regressions:
                    regression_detected = True
                    exit_code = EXIT_EVAL_FAILED

    except Exception as exc:
        error = f"Runtime error: {exc}"
        exit_code = EXIT_RUNTIME_ERROR

    output = format_json_output(
        summary, error=error, regression_detected=regression_detected
    )
    write_output(output, args.output)
    return exit_code


def write_output(output: dict[str, Any], output_path: str | None) -> None:
    """Write JSON output to file or stdout."""
    json_str = json.dumps(output, indent=2)

    if output_path:
        Path(output_path).write_text(json_str)
        print(f"Results written to: {output_path}", file=sys.stderr)
    else:
        print(json_str)


def main() -> None:
    """Main entry point for cron/CI evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Cron/CI-friendly evaluation runner (JSON output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config/eng_smoke.yaml
  %(prog)s config/eng_smoke.yaml --output results.json
  %(prog)s config/system_integrity.yaml --check-regression
  %(prog)s config/eng_smoke.yaml -g http://gateway:8000 -o /tmp/eval.json
        """,
    )
    parser.add_argument(
        "config",
        help="Path to the suite configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "-g", "--gateway-url",
        default=None,
        help="Gateway base URL (default: GATEWAY_BASE_URL env)",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        dest="check_regression",
        help="Compare against baseline and fail on critical regressions",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(run_cron_evaluation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
