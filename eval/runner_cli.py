#!/usr/bin/env python3
"""CLI entrypoint for running evaluation suites."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from uuid import UUID

from eval.database import get_supabase_client
from eval.env import load_project_env
from eval.controller import EvaluationController
from eval.models import EvalRunSummary, EvalSuiteConfig
from eval.regression import RegressionDetector, format_regression_report

load_project_env()


EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_EVAL_FAILED = 2
EXIT_RUNTIME_ERROR = 3


def load_suite_config(config_path: str) -> EvalSuiteConfig:
    """Load suite configuration from YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        EvalSuiteConfig parsed from the file

    Raises:
        ValueError: If the file format is unsupported or invalid
    """
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


def format_summary(summary: EvalRunSummary, verbose: bool = False) -> str:
    """Format an eval run summary for display.

    Args:
        summary: The evaluation run summary
        verbose: Whether to include detailed per-task results

    Returns:
        Formatted string for display
    """
    run = summary.run
    metrics = run.metrics
    lines = [
        f"Evaluation Run: {run.name} v{run.version}",
        f"Run ID: {run.id}",
        f"Status: {run.status.value}",
        f"Result: {'PASSED' if summary.passed else 'FAILED'}",
        "",
        "Metrics:",
        f"  Total Tasks: {metrics.total_tasks}",
        f"  Completed: {metrics.completed_tasks}",
        f"  Failed: {metrics.failed_tasks}",
        f"  Timed Out: {metrics.timed_out_tasks}",
        f"  Average Score: {metrics.average_score:.3f}",
        f"  Success Rate: {metrics.success_rate:.1%}",
        f"  Total Cost: ${metrics.total_cost_usd:.4f}",
        f"  Avg Duration: {metrics.average_duration_seconds:.2f}s",
    ]

    if metrics.by_work_area:
        lines.append("")
        lines.append("By Work Area:")
        for area, data in metrics.by_work_area.items():
            lines.append(
                f"  {area}: score={data.get('average_score', 0):.3f}, "
                f"success={data.get('success_rate', 0):.1%}, "
                f"count={int(data.get('count', 0))}"
            )

    if metrics.by_cost_tier:
        lines.append("")
        lines.append("By Cost Tier:")
        for tier, data in metrics.by_cost_tier.items():
            lines.append(
                f"  {tier}: score={data.get('average_score', 0):.3f}, "
                f"cost=${data.get('average_cost_usd', 0):.4f}, "
                f"count={int(data.get('count', 0))}"
            )

    if verbose and summary.results:
        lines.append("")
        lines.append("Task Results:")
        for result in summary.results:
            status_icon = (
                "✓" if result.status.value == "completed" and result.score >= 0.8
                else "✗" if result.status.value in ("failed", "timed_out")
                else "○"
            )
            lines.append(
                f"  {status_icon} {result.eval_task_id}: "
                f"status={result.status.value}, "
                f"score={result.score:.3f}, "
                f"duration={result.duration_seconds or 0:.2f}s"
            )

    return "\n".join(lines)


def format_json_summary(summary: EvalRunSummary) -> str:
    """Format an eval run summary as JSON.

    Args:
        summary: The evaluation run summary

    Returns:
        JSON string representation
    """
    data = {
        "run": {
            "id": str(summary.run.id),
            "name": summary.run.name,
            "version": summary.run.version,
            "status": summary.run.status.value,
            "total_tasks": summary.run.total_tasks,
            "created_at": summary.run.created_at.isoformat(),
            "finished_at": (
                summary.run.finished_at.isoformat()
                if summary.run.finished_at
                else None
            ),
        },
        "metrics": summary.run.metrics.model_dump(),
        "passed": summary.passed,
        "results": [
            {
                "eval_task_id": r.eval_task_id,
                "gateway_task_id": str(r.gateway_task_id) if r.gateway_task_id else None,
                "status": r.status.value,
                "score": r.score,
                "duration_seconds": r.duration_seconds,
                "errors": r.errors if r.errors else None,
            }
            for r in summary.results
        ],
    }
    return json.dumps(data, indent=2)


async def run_evaluation(args: argparse.Namespace) -> int:
    """Execute an evaluation suite.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        config = load_suite_config(args.config)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.concurrency:
        config.max_concurrency = args.concurrency

    if args.timeout:
        config.task_timeout_seconds = args.timeout

    try:
        supabase = get_supabase_client()
        controller = EvaluationController(
            supabase,
            gateway_base_url=args.gateway_url or os.environ.get("GATEWAY_BASE_URL"),
        )

        # Get baseline for regression comparison if requested
        baseline_run_id = None
        if getattr(args, 'compare_baseline', False):
            detector = RegressionDetector(supabase)
            baseline_run_id = detector.get_latest_baseline(config.name)
            if baseline_run_id:
                print(f"Will compare against baseline run: {str(baseline_run_id)[:8]}...")
            else:
                print("No baseline found for regression comparison")

        print(f"Starting evaluation: {config.name} v{config.version}")
        print(f"Questions: {len(config.question_ids)}, Concurrency: {config.max_concurrency}")
        print()

        summary = await controller.run_eval_suite(config)

        if args.json:
            print(format_json_summary(summary))
        else:
            print(format_summary(summary, verbose=args.verbose))

        # Run regression comparison if baseline exists
        if baseline_run_id and summary.run.id:
            print()
            print("Running regression comparison...")
            detector = RegressionDetector(supabase)
            report = detector.compare_runs(baseline_run_id, summary.run.id)
            print()
            print(format_regression_report(report))
            
            # Fail if critical regressions detected
            if report.has_critical_regressions:
                print("\n❌ Critical regressions detected - marking as failed")
                return EXIT_EVAL_FAILED

        return EXIT_SUCCESS if summary.passed else EXIT_EVAL_FAILED

    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
        else:
            print(f"Runtime error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def run_comparison(args: argparse.Namespace) -> int:
    """Compare two eval runs for regression detection.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for no regressions, non-zero for regressions found)
    """
    try:
        supabase = get_supabase_client()
        detector = RegressionDetector(supabase)
        
        baseline_id = UUID(args.baseline)
        current_id = UUID(args.current)
        
        report = detector.compare_runs(baseline_id, current_id)
        
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_regression_report(report))
        
        if report.has_critical_regressions:
            return EXIT_EVAL_FAILED
        elif report.has_regressions:
            return EXIT_CONFIG_ERROR  # Warning level
        return EXIT_SUCCESS
        
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run evaluation suites against the Task Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command (default behavior)
    run_parser = subparsers.add_parser(
        "run",
        help="Run an evaluation suite",
        epilog="""
Examples:
  %(prog)s config/eng_smoke.yaml
  %(prog)s config/full_suite.yaml --concurrency 10 --verbose
  %(prog)s config/quick_test.json --json --timeout 60
  %(prog)s config/eng_smoke.yaml --compare-baseline
        """,
    )
    run_parser.add_argument(
        "config",
        help="Path to the suite configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=None,
        help="Override max concurrency from config",
    )
    run_parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=None,
        help="Override per-task timeout (seconds) from config",
    )
    run_parser.add_argument(
        "-g", "--gateway-url",
        default=None,
        help="Gateway base URL (default: GATEWAY_BASE_URL env or http://localhost:8000)",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed per-task results",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    run_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        dest="compare_baseline",
        help="Compare against latest baseline run and detect regressions",
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two eval runs for regression detection",
        epilog="""
Examples:
  %(prog)s abc123 def456
  %(prog)s abc123 def456 --json
        """,
    )
    compare_parser.add_argument(
        "baseline",
        help="Baseline run ID to compare against",
    )
    compare_parser.add_argument(
        "current",
        help="Current run ID to check for regressions",
    )
    compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    
    # Handle no subcommand (backward compatibility: treat first arg as config)
    if args.command is None:
        # Check if there's a positional argument that looks like a config path
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Re-parse with "run" prepended
            sys.argv.insert(1, "run")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(EXIT_CONFIG_ERROR)
    
    if args.command == "run":
        exit_code = asyncio.run(run_evaluation(args))
    elif args.command == "compare":
        exit_code = run_comparison(args)
    else:
        parser.print_help()
        exit_code = EXIT_CONFIG_ERROR
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
