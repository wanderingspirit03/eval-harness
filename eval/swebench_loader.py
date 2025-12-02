"""SWE-bench Lite dataset loader for the eval harness.

This module loads SWE-bench Lite from HuggingFace and seeds the eval_questions
table with instances that can be evaluated using our existing harness.

Usage:
    python -m agent_cluster.eval.swebench_loader --limit 10 --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import os

from datasets import load_dataset
from supabase import create_client


def get_supabase_client():
    """Get Supabase client from environment."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def load_swebench_lite(split: str = "dev", limit: int | None = None) -> list[dict[str, Any]]:
    """Load SWE-bench Lite dataset from HuggingFace.
    
    Args:
        split: Dataset split ('dev' has 23 instances, 'test' has 300)
        limit: Maximum number of instances to load
    
    Returns:
        List of SWE-bench instances
    """
    print(f"Loading SWE-bench Lite ({split} split)...")
    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split=split)
    
    instances = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        instances.append(dict(item))
    
    print(f"Loaded {len(instances)} instances")
    return instances


def transform_to_eval_question(instance: dict[str, Any]) -> dict[str, Any]:
    """Transform a SWE-bench instance into our eval_questions format.
    
    Args:
        instance: SWE-bench instance with repo, problem_statement, etc.
    
    Returns:
        Dict ready to insert into eval_questions table
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    
    # Parse the tests that need to pass
    fail_to_pass = json.loads(instance["FAIL_TO_PASS"]) if instance.get("FAIL_TO_PASS") else []
    pass_to_pass = json.loads(instance["PASS_TO_PASS"]) if instance.get("PASS_TO_PASS") else []
    
    # Build the prompt for the agent
    prompt = f"""You are solving a GitHub issue from the {repo} repository.

## Problem Statement
{problem_statement}

## Task
1. Clone the repository: git clone https://github.com/{repo}.git
2. Checkout the base commit: git checkout {base_commit}
3. Analyze the problem and implement a fix
4. Run the failing tests to verify your fix works

## Tests to Fix (must pass after your changes)
{json.dumps(fail_to_pass, indent=2)}

## Existing Tests (must still pass)
These tests should continue to pass: {len(pass_to_pass)} tests

Implement the fix and report which tests pass after your changes."""

    # Build metadata for test_suite scoring
    metadata = {
        "suite": "swebench_lite",
        "repo": repo,
        "base_commit": base_commit,
        "instance_id": instance_id,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass[:10],  # Limit for storage
        "version": instance.get("version", ""),
        "gold_patch": instance.get("patch", "")[:2000],  # Truncate for storage
        "test_patch": instance.get("test_patch", "")[:2000],
    }
    
    return {
        "id": f"swebench_{instance_id.replace('/', '_').replace('-', '_').lower()}",
        "prompt": prompt,
        "category": "eng",
        "work_area": "swebench",
        "cost_tier": "expensive",  # These take time and resources
        "expected_output": None,  # No exact match - use test_suite scoring
        "scoring_mode": "test_suite",
        "metadata": metadata,
    }


def seed_eval_questions(
    instances: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    replace: bool = False,
) -> int:
    """Insert SWE-bench instances into the eval_questions table.
    
    Args:
        instances: List of SWE-bench instances
        dry_run: If True, only print what would be inserted
        replace: If True, replace existing questions with same ID
    
    Returns:
        Number of questions inserted/updated
    """
    supabase = get_supabase_client()
    count = 0
    
    for instance in instances:
        question = transform_to_eval_question(instance)
        
        if dry_run:
            print(f"[DRY RUN] Would insert: {question['id']}")
            print(f"  Repo: {instance['repo']}")
            print(f"  Tests to fix: {len(json.loads(instance['FAIL_TO_PASS']))}")
            count += 1
            continue
        
        try:
            if replace:
                # Upsert - insert or update
                response = supabase.table("eval_questions").upsert(question).execute()
            else:
                # Check if exists
                existing = supabase.table("eval_questions").select("id").eq("id", question["id"]).execute()
                if existing.data:
                    print(f"Skipping existing: {question['id']}")
                    continue
                response = supabase.table("eval_questions").insert(question).execute()
            
            if response.data:
                print(f"Inserted: {question['id']}")
                count += 1
            else:
                print(f"Failed to insert: {question['id']}")
        except Exception as e:
            print(f"Error inserting {question['id']}: {e}")
    
    return count


def list_swebench_questions() -> None:
    """List existing SWE-bench questions in the database."""
    supabase = get_supabase_client()
    
    response = supabase.table("eval_questions").select("id, category, work_area, metadata").eq("work_area", "swebench").execute()
    
    print(f"\n{'='*60}")
    print("Existing SWE-bench questions in eval_questions:")
    print(f"{'='*60}")
    
    if not response.data:
        print("No SWE-bench questions found")
        return
    
    for q in response.data:
        meta = q.get("metadata") or {}
        print(f"  {q['id']}")
        print(f"    Repo: {meta.get('repo', 'N/A')}")
        print(f"    Tests: {len(meta.get('fail_to_pass', []))} to fix")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load SWE-bench Lite into eval_questions table"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split (dev=23 instances, test=300 instances)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to load",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without inserting",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing questions with same ID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing SWE-bench questions in database",
    )
    
    args = parser.parse_args(argv)
    
    if args.list:
        list_swebench_questions()
        return 0
    
    instances = load_swebench_lite(split=args.split, limit=args.limit)
    
    if not instances:
        print("No instances loaded")
        return 1
    
    count = seed_eval_questions(instances, dry_run=args.dry_run, replace=args.replace)
    
    print(f"\n{'='*60}")
    print(f"Summary: {'Would insert' if args.dry_run else 'Inserted'} {count} questions")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
