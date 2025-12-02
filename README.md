# Evaluation Harness

An evaluation harness that extends the Task Gateway Service to measure and score AI agent behavior. The harness acts as an internal client of the gateway, exercising the same `/api/v1/tasks` interface used by real agents and storing structured results in Supabase for regression analysis.

## Features

- **REST Client**: Submits tasks to gateway `/api/v1/tasks` endpoint
- **Polling**: Polls for completion and extracts `result_metadata`
- **Scoring Engine**: Multiple scoring modes (exact match, test suite, rubric/LLM-as-judge)
- **Result Storage**: Stores execution traces, costs, and timing data in Supabase
- **Batch Execution**: Supports batch submission with concurrency limits
- **Regression Detection**: Compare runs and detect performance regressions

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
```bash
# Install all dependencies
uv sync

# Or install with optional dependencies
uv sync --extra tracing  # For Laminar tracing support
uv sync --extra datasets  # For SWE-bench dataset loading
uv sync --extra all  # All optional dependencies
```

### Upgrade dependencies
```bash
# Upgrade all dependencies to latest compatible versions
uv sync --upgrade

# Upgrade specific package
uv sync --upgrade-package supabase
```

## Quick Start

1. **Set up environment variables:**
```bash
export SUPABASE_URL=http://localhost:54321  # or your Supabase URL
export SUPABASE_KEY=your_supabase_key
export GATEWAY_BASE_URL=http://localhost:8000
```

2. **Seed evaluation questions in Supabase:**
```sql
INSERT INTO eval_questions (id, prompt, category, work_area, cost_tier, expected_output, scoring_mode, metadata)
VALUES
  (
    'test_exact_match',
    'What is 2+2?',
    'test', 'math', 'cheap', '"4"', 'exact_match', '{"suite": "test"}'::jsonb
  );
```

3. **Run an evaluation suite:**
```bash
# Using uv to run
uv run python -m eval.runner_cli run config/eng_smoke.yaml --gateway-url http://localhost:8000 --verbose

# Or activate the virtual environment first
source .venv/bin/activate  # Created by uv sync
python -m eval.runner_cli run config/eng_smoke.yaml --gateway-url http://localhost:8000 --verbose
```

## Configuration

Create a YAML configuration file for your evaluation suite:

```yaml
name: my_suite
version: "1.0.0"
question_ids:
  - question_id_1
  - question_id_2
max_concurrency: 5
task_timeout_seconds: 300.0
poll_interval_seconds: 2.0
success_threshold: 0.8
```

## Scoring Modes

- **exact_match**: Compares agent output to expected output
- **test_suite**: Evaluates based on test results (tests_passed/tests_failed)
- **rubric**: Uses LLM-as-judge for open-ended evaluations

## Database Schema

The harness requires the following Supabase tables:
- `eval_questions`: Source of eval questions and expected outputs
- `eval_runs`: Metadata for each evaluation execution
- `eval_results`: Per-task evaluation outcomes

See the main README.md for detailed schema definitions.

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Upgrade all dependencies
uv sync --upgrade
```

## License

See LICENSE file for details.
