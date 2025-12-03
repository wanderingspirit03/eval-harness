from __future__ import annotations

from typing import List

import pytest

from eval.controller import EvaluationController
from eval.models import CostTier, EvalQuestion, EvalSuiteConfig


class _FakeEvalSupabaseClient:
    def __init__(self, questions: List[EvalQuestion]):
        self._questions = questions
        self.inserted_tasks: list[dict[str, object]] | None = None
        self.created_results: list | None = None
        self.created_run = None

    def get_questions_by_ids(self, question_ids: list[str]) -> list[EvalQuestion]:
        return self._questions

    def create_eval_run(self, run):
        self.created_run = run

    def bulk_insert_tasks(self, tasks: list[dict[str, object]]) -> None:
        self.inserted_tasks = tasks

    def bulk_create_eval_results(self, results: list) -> None:
        self.created_results = results


class _StubSupabase:
    def table(self, name: str):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_seed_eval_suite_sets_eval_metadata() -> None:
    question = EvalQuestion(id="q1", prompt="do something", work_area="eng", cost_tier=CostTier.CHEAP)
    fake_client = _FakeEvalSupabaseClient([question])

    controller = EvaluationController(supabase=_StubSupabase())
    controller._supabase = fake_client  # override with fake

    suite = EvalSuiteConfig(name="suite", question_ids=["q1"], metadata={})

    await controller.seed_eval_suite(suite)

    assert fake_client.inserted_tasks is not None
    metadata = fake_client.inserted_tasks[0]["result_metadata"]
    assert metadata["source"] == "eval"
    assert metadata["work_area"] == "eng"
    assert metadata["cost_tier"] == CostTier.CHEAP.value

    assert fake_client.created_results is not None
    assert fake_client.created_results[0].started_at is None
