"""Scoring engine for evaluation results."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from eval.models import EvalQuestion, ScoringMode
from eval.observability import traced_span
from lmnr import observe


class ScoringError(Exception):
    """Base exception for scoring errors."""


class ScoreResult:
    """Result of a scoring operation."""

    def __init__(
        self,
        score: float,
        raw_score: dict[str, Any],
        *,
        mode: ScoringMode,
        rationale: str | None = None,
    ):
        self.score = max(0.0, min(1.0, score))
        self.raw_score = raw_score
        self.mode = mode
        self.rationale = rationale

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "mode": self.mode.value,
            "rationale": self.rationale,
            **self.raw_score,
        }


class Scorer(ABC):
    """Abstract base class for scorers."""

    @abstractmethod
    async def score(
        self,
        question: EvalQuestion,
        actual_result: dict[str, Any],
    ) -> ScoreResult:
        """Score an actual result against expected output.

        Args:
            question: The eval question with expected output/rubric
            actual_result: The actual result from the gateway task

        Returns:
            ScoreResult with normalized score and details
        """


class ExactMatchScorer(Scorer):
    """Scorer for exact match comparisons."""

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    async def score(
        self,
        question: EvalQuestion,
        actual_result: dict[str, Any],
    ) -> ScoreResult:
        """Compare actual output to expected output exactly.

        Extracts 'answer_text' or 'result' from actual_result and compares
        to question.expected_output.
        """
        with traced_span("eval.scoring.exact_match", input={"question_id": question.id}):
            expected = question.expected_output
            actual = actual_result.get("result_metadata", {}).get("answer_text")
            if actual is None:
                actual = actual_result.get("result", "")

            if expected is None:
                return ScoreResult(
                    score=0.0,
                    raw_score={"error": "No expected output defined"},
                    mode=ScoringMode.EXACT_MATCH,
                )

            expected_str = self._normalize(str(expected))
            actual_str = self._normalize(str(actual))

            match = expected_str == actual_str
            return ScoreResult(
                score=1.0 if match else 0.0,
                raw_score={
                    "expected": expected_str,
                    "actual": actual_str,
                    "match": match,
                },
                mode=ScoringMode.EXACT_MATCH,
                rationale="Exact match" if match else "No match",
            )

    def _normalize(self, text: str) -> str:
        if not self.normalize:
            return text
        result = text.strip()
        result = re.sub(r"\s+", " ", result)
        if not self.case_sensitive:
            result = result.lower()
        return result


class TestSuiteScorer(Scorer):
    """Scorer for test suite results.
    
    Supports multiple ways to extract test results:
    1. Structured metadata (tests_passed, tests_failed)
    2. test_results list
    3. exit_code fallback
    4. Text parsing from agent output (pytest, unittest patterns)
    """

    # Patterns to extract test results from text output
    TEST_PATTERNS = [
        # pytest style: "3 passed, 2 failed"
        r"(\d+)\s+passed(?:,?\s+(\d+)\s+failed)?",
        # pytest summary: "===== 3 passed in 1.23s ====="
        r"=+\s*(\d+)\s+passed(?:,?\s*(\d+)\s+failed)?.*=+",
        # unittest style: "Ran 5 tests... OK" or "Ran 5 tests... FAILED (failures=2)"
        r"Ran\s+(\d+)\s+tests?.*?(OK|FAILED)",
        # Generic: "Tests: 3 passed, 1 failed"
        r"[Tt]ests?:?\s*(\d+)\s+passed(?:,?\s*(\d+)\s+failed)?",
        # "All 5 tests passed"
        r"[Aa]ll\s+(\d+)\s+tests?\s+passed",
        # "(N tests)" or "N tests" at end of line - implies all passed
        r"\((\d+)\s+tests?\)",
        # "all tests in that file passed" followed by count like "(30 tests)"
        r"all\s+tests.*passed.*\((\d+)\s+tests?\)",
        # pytest dots: "......" (30 tests) - count the number in parentheses
        r"\.{3,}.*\((\d+)\s+tests?\)",
    ]

    def _parse_test_results_from_text(self, text: str) -> tuple[int, int] | None:
        """Try to extract test pass/fail counts from text output.
        
        Args:
            text: The agent's text output
            
        Returns:
            Tuple of (passed, failed) or None if not found
        """
        if not text:
            return None
            
        for pattern in self.TEST_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                groups = match.groups()
                passed = int(groups[0]) if groups[0] else 0
                
                # Handle different pattern formats
                if len(groups) >= 2 and groups[1]:
                    if groups[1] in ("OK", "FAILED"):
                        # unittest style - if OK, 0 failures
                        failed = 0 if groups[1] == "OK" else 1
                    else:
                        failed = int(groups[1])
                else:
                    failed = 0
                    
                return (passed, failed)
        
        return None

    async def score(
        self,
        question: EvalQuestion,
        actual_result: dict[str, Any],
    ) -> ScoreResult:
        """Score based on test results in the actual output.

        Looks for test_results, tests_passed, tests_failed, exit_code,
        or parses test output from the agent's text result.
        """
        with traced_span("eval.scoring.test_suite", input={"question_id": question.id}):
            metadata = actual_result.get("result_metadata", {})
            result_text = actual_result.get("result", "")
            answer_text = metadata.get("answer_text", "")

            tests_passed = metadata.get("tests_passed", 0)
            tests_failed = metadata.get("tests_failed", 0)
            exit_code = metadata.get("exit_code")
            scoring_method = "structured"

            # Check for test_results list
            test_results = metadata.get("test_results", [])
            if test_results and isinstance(test_results, list):
                tests_passed = sum(1 for t in test_results if t.get("passed", False))
                tests_failed = len(test_results) - tests_passed
                scoring_method = "test_results_list"

            # If no structured data, try to parse from text
            total_tests = tests_passed + tests_failed
            if total_tests == 0:
                # Try parsing from answer_text first, then result
                parsed = self._parse_test_results_from_text(answer_text)
                if parsed is None:
                    parsed = self._parse_test_results_from_text(result_text)
                
                if parsed:
                    tests_passed, tests_failed = parsed
                    total_tests = tests_passed + tests_failed
                    scoring_method = "text_parsing"

            # Calculate score
            total_tests = tests_passed + tests_failed
            if total_tests == 0:
                # Fallback to exit code
                if exit_code is not None:
                    score = 1.0 if exit_code == 0 else 0.0
                    return ScoreResult(
                        score=score,
                        raw_score={
                            "exit_code": exit_code,
                            "scoring_method": "exit_code",
                        },
                        mode=ScoringMode.TEST_SUITE,
                        rationale="Scored by exit code" if exit_code == 0 else "Non-zero exit code",
                    )
                return ScoreResult(
                    score=0.0,
                    raw_score={"error": "No test results found in metadata or text output"},
                    mode=ScoringMode.TEST_SUITE,
                )

            score = tests_passed / total_tests

            return ScoreResult(
                score=score,
                raw_score={
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                    "total_tests": total_tests,
                    "exit_code": exit_code,
                    "scoring_method": scoring_method,
                },
                mode=ScoringMode.TEST_SUITE,
                rationale=f"{tests_passed}/{total_tests} tests passed",
            )


class RubricScorer(Scorer):
    """Scorer using LLM-as-judge with a rubric.

    Uses OpenAI's gpt-5.1-2025-11-13 model by default. Supports:
    - Cost tracking for judge API calls
    - Weighted dimension scoring
    - Configurable retry logic for API failures
    """

    DEFAULT_JUDGE_MODEL = "gpt-5.1-2025-11-13"
    DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    MAX_SCORE = 10
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_url = api_url or os.environ.get("JUDGE_API_URL", self.DEFAULT_OPENAI_URL)
        self.model = model or os.environ.get("JUDGE_MODEL", self.DEFAULT_JUDGE_MODEL)
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("JUDGE_MAX_RETRIES", self.DEFAULT_MAX_RETRIES))
        self.retry_delay = retry_delay if retry_delay is not None else float(os.environ.get("JUDGE_RETRY_DELAY", self.DEFAULT_RETRY_DELAY))

    async def score(
        self,
        question: EvalQuestion,
        actual_result: dict[str, Any],
    ) -> ScoreResult:
        """Score using an LLM judge with the question's rubric.

        Constructs a judge prompt with the original task, answer, and rubric,
        then calls the judge API to get a score and rationale. Tracks API
        costs and supports weighted dimension scoring.
        """
        with traced_span("eval.scoring.rubric", input={"question_id": question.id}):
            if not self.api_key:
                return ScoreResult(
                    score=0.0,
                    raw_score={"error": "No JUDGE_API_KEY or OPENAI_API_KEY configured"},
                    mode=ScoringMode.RUBRIC,
                )

            rubric = question.rubric or {}
            answer = actual_result.get("result_metadata", {}).get("answer_text")
            if answer is None:
                answer = actual_result.get("result", "")

            judge_prompt = self._build_judge_prompt(
                task_prompt=question.prompt,
                answer=str(answer),
                rubric=rubric,
            )

            try:
                judge_response, usage_data = await self._call_judge_api_with_retry(judge_prompt)

                # Calculate cost from usage data
                judge_cost = self._calculate_judge_cost(usage_data)

                # Get dimension scores and weights
                dimension_scores = judge_response.get("dimension_scores", {})
                weights = rubric.get("weights", {})

                # Calculate final score (weighted if weights provided, else raw judge score)
                if dimension_scores and weights:
                    score = self._compute_weighted_score(dimension_scores, weights)
                else:
                    score = judge_response.get("score", 0) / self.MAX_SCORE

                return ScoreResult(
                    score=score,
                    raw_score={
                        "judge_score": judge_response.get("score"),
                        "max_score": self.MAX_SCORE,
                        "dimension_scores": dimension_scores,
                        "weighted_score": score if weights else None,
                        "weights_applied": bool(weights),
                        "model": self.model,
                        "judge_cost": judge_cost,
                    },
                    mode=ScoringMode.RUBRIC,
                    rationale=judge_response.get("rationale"),
                )
            except Exception as exc:
                return ScoreResult(
                    score=0.0,
                    raw_score={"error": str(exc), "model": self.model},
                    mode=ScoringMode.RUBRIC,
                    rationale=f"Judge API error: {exc}",
                )

    def _compute_weighted_score(
        self,
        dimension_scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Compute weighted average of dimension scores.

        Args:
            dimension_scores: Dict of dimension name to score (1-10)
            weights: Dict of dimension name to weight (any positive number)

        Returns:
            Normalized score between 0 and 1
        """
        if not dimension_scores:
            return 0.0

        total_weighted = 0.0
        total_weight = 0.0

        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 1.0)
            total_weighted += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize: dimension scores are 1-10, final score should be 0-1
        return (total_weighted / total_weight) / self.MAX_SCORE

    def _calculate_judge_cost(self, usage_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate cost for judge API call using pricing module.

        Args:
            usage_data: OpenAI usage object with prompt_tokens, completion_tokens, etc.

        Returns:
            Dict with model, tokens, and USD cost
        """
        from eval.pricing import calculate_cost

        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
        cached_tokens = usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        cost_data = calculate_cost(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

        return {
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "usd": cost_data["usd"],
        }

    def _build_judge_prompt(
        self,
        task_prompt: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> str:
        rubric_text = json.dumps(rubric, indent=2) if rubric else "No specific rubric provided."

        # Extract criteria list for clearer instructions
        criteria = rubric.get("criteria", [])
        criteria_text = ", ".join(criteria) if criteria else "overall quality"

        return f"""You are an expert evaluator. Score the following answer to a task.

## Task
{task_prompt}

## Answer
{answer}

## Rubric
{rubric_text}

## Instructions
Evaluate the answer based on the rubric criteria: {criteria_text}.

For each criterion, provide a score from 1-10 where:
- 1-3: Poor - Does not meet expectations
- 4-6: Acceptable - Partially meets expectations
- 7-8: Good - Meets expectations
- 9-10: Excellent - Exceeds expectations

Provide your response in the following JSON format:
{{
    "score": <integer 1-10, overall score>,
    "dimension_scores": {{{", ".join(f'"{c}": <score>' for c in criteria) if criteria else '"quality": <score>'}}},
    "rationale": "<brief explanation of scoring>"
}}

Only output valid JSON, nothing else."""

    async def _call_judge_api_with_retry(
        self,
        prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call the judge API with retry logic.

        Args:
            prompt: The judge prompt

        Returns:
            Tuple of (parsed response, usage data)

        Raises:
            ScoringError: If all retries fail
        """
        import asyncio

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self._call_judge_api(prompt)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    # Exponential backoff: delay * (2 ^ attempt)
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        raise ScoringError(f"Judge API failed after {self.max_retries} retries: {last_error}")

    async def _call_judge_api(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call the judge LLM API.

        Args:
            prompt: The judge prompt

        Returns:
            Tuple of (parsed JSON response, usage data)

        Raises:
            ScoringError: If the API call fails or response is invalid
        """
        with traced_span("eval.scoring.judge_api_call", input={"model": self.model}):
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Extract usage data for cost tracking
                usage_data = data.get("usage", {})

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                try:
                    # With JSON mode, response should be valid JSON
                    parsed = json.loads(content)
                    return parsed, usage_data
                except json.JSONDecodeError:
                    # Fallback: try to extract JSON from response
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group()), usage_data
                    raise ScoringError(f"Failed to parse judge response: {content}")


class ScoringEngine:
    """Main scoring engine that dispatches to appropriate scorer."""

    def __init__(self):
        self._scorers: dict[ScoringMode, Scorer] = {
            ScoringMode.EXACT_MATCH: ExactMatchScorer(),
            ScoringMode.TEST_SUITE: TestSuiteScorer(),
            ScoringMode.RUBRIC: RubricScorer(),
        }

    def register_scorer(self, mode: ScoringMode, scorer: Scorer) -> None:
        """Register a custom scorer for a scoring mode."""
        self._scorers[mode] = scorer

    async def score(
        self,
        question: EvalQuestion,
        actual_result: dict[str, Any],
    ) -> ScoreResult:
        """Score an actual result using the appropriate scorer.

        Args:
            question: The eval question with scoring mode and expected output
            actual_result: The actual task result from the gateway

        Returns:
            ScoreResult with normalized score and details

        Raises:
            ScoringError: If no scorer is available for the scoring mode
        """
        with traced_span(
            "eval.scoring.score",
            input={"question_id": question.id, "mode": question.scoring_mode.value},
        ):
            scorer = self._scorers.get(question.scoring_mode)
            if scorer is None:
                raise ScoringError(f"No scorer available for mode: {question.scoring_mode}")

            return await scorer.score(question, actual_result)


__all__ = [
    "ExactMatchScorer",
    "RubricScorer",
    "ScoreResult",
    "Scorer",
    "ScoringEngine",
    "ScoringError",
    "TestSuiteScorer",
]
