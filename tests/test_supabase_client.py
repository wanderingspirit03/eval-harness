from datetime import datetime, timezone
from uuid import uuid4

from eval.models import EvalResult, EvalStatus
from eval.supabase_client import EvalSupabaseClient


class _DummyTable:
    def __init__(self, capture: dict[str, object]):
        self._capture = capture

    def update(self, payload: dict[str, object]):
        self._capture["payload"] = payload
        return self

    def eq(self, column: str, value: str):
        self._capture["filter"] = (column, value)
        return self

    def execute(self):
        class _Response:
            error = None

        self._capture["executed"] = True
        return _Response()


class _DummySupabase:
    def __init__(self):
        self.capture: dict[str, object] = {}

    def table(self, name: str):
        self.capture["table"] = name
        return _DummyTable(self.capture)


def test_normalize_task_join_handles_dict_and_list() -> None:
    row = {"status": "completed"}
    assert EvalSupabaseClient._normalize_task_join(row) == row

    wrapped = [row]
    assert EvalSupabaseClient._normalize_task_join(wrapped) == row

    assert EvalSupabaseClient._normalize_task_join(None) is None


def test_update_eval_result_persists_payload() -> None:
    dummy = _DummySupabase()
    client = EvalSupabaseClient(dummy)  # type: ignore[arg-type]

    result = EvalResult(
        eval_run_id=uuid4(),
        eval_task_id="task-1",
        gateway_task_id=None,
        status=EvalStatus.COMPLETED.value,
        finished_at=datetime.now(timezone.utc),
        duration_seconds=2.5,
        cost={"usd": 1.0},
    )

    client.update_eval_result(result, agent_output="answer")

    assert dummy.capture["table"] == "eval_results"
    assert dummy.capture["filter"] == ("id", str(result.id))

    payload = dummy.capture["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "completed"
    assert payload["agent_output"] == "answer"
    assert payload["metadata"] == {}
    assert dummy.capture["executed"] is True
