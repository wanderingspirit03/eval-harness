from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from eval.agent_logs import AgentConversationArtifacts, derive_agent_conversation


class _StubEvalClient:
    def __init__(self) -> None:
        self.task_rows: list[dict[str, object]] = []
        self.task_by_id: dict[str, dict[str, object]] = {}
        self.agents: dict[str, dict[str, object]] = {}
        self.manager_agents: dict[str, list[dict[str, object]]] = {}
        self.messages: dict[str, list[dict[str, object]]] = {}

    def get_task_by_id(self, task_id: str) -> dict[str, object] | None:
        return self.task_by_id.get(task_id)

    def get_tasks_for_eval(self, eval_run_id: UUID, eval_task_id: str) -> list[dict[str, object]]:
        return self.task_rows

    def get_agents_by_ids(self, agent_ids: list[str]) -> list[dict[str, object]]:
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]

    def get_engineers_for_manager(self, manager_id: str) -> list[dict[str, object]]:
        return self.manager_agents.get(manager_id, [])

    def get_messages_for_agent(self, agent_id: str) -> list[dict[str, object]]:
        return self.messages.get(agent_id, [])


@pytest.mark.asyncio
async def test_derive_agent_conversation_prefers_latest_assistant_message() -> None:
    stub = _StubEvalClient()
    task_id = uuid4()
    stub.task_by_id[str(task_id)] = {
        "task_id": str(task_id),
        "assigned_agent_ids": ["eng-1"],
        "manager_id": "mgr-1",
    }
    stub.task_rows = [stub.task_by_id[str(task_id)]]

    stub.agents["eng-1"] = {
        "agent_id": "eng-1",
        "agent_type": "engineer",
        "status": "completed",
        "completed_at": "2025-12-03T12:00:00Z",
        "result": {"answer_text": "unused"},
    }

    stub.messages["eng-1"] = [
        {
            "agent_id": "eng-1",
            "turn_number": 0,
            "role": "user",
            "content": {
                "content": [
                    {"block_type": "TextBlock", "data": {"text": "Input context"}},
                ]
            },
        },
        {
            "agent_id": "eng-1",
            "turn_number": 1,
            "role": "assistant",
            "content": {
                "content": [
                    {
                        "block_type": "TextBlock",
                        "data": {"text": "First draft"},
                    },
                    {
                        "block_type": "TextBlock",
                        "data": {"text": "Final Answer: DONE"},
                    },
                ]
            },
        },
    ]

    artifacts = await derive_agent_conversation(
        stub, uuid4(), "eval-task", gateway_task_id=task_id
    )

    assert artifacts.answer_text == "DONE"
    assert artifacts.engineer_status == "completed"
    assert artifacts.log is not None
    assert len(artifacts.log) == 2


@pytest.mark.asyncio
async def test_derive_agent_conversation_falls_back_to_agent_result_text() -> None:
    stub = _StubEvalClient()
    task_id = uuid4()
    stub.task_by_id[str(task_id)] = {
        "task_id": str(task_id),
        "assigned_agent_ids": ["eng-2"],
        "manager_id": "mgr-1",
    }
    stub.task_rows = [stub.task_by_id[str(task_id)]]

    stub.agents["eng-2"] = {
        "agent_id": "eng-2",
        "agent_type": "engineer",
        "status": "completed",
        "completed_at": "2025-12-03T13:00:00Z",
        "result": {"answer_text": "Fallback Answer"},
    }

    stub.messages["eng-2"] = []

    artifacts = await derive_agent_conversation(
        stub, uuid4(), "eval-task", gateway_task_id=task_id
    )

    assert artifacts.answer_text == "Fallback Answer"
    assert artifacts.log is None


@pytest.mark.asyncio
async def test_derive_agent_conversation_no_terminal_engineer_returns_empty() -> None:
    stub = _StubEvalClient()
    task_id = uuid4()
    stub.task_by_id[str(task_id)] = {
        "task_id": str(task_id),
        "assigned_agent_ids": ["eng-3"],
        "manager_id": "mgr-1",
    }
    stub.task_rows = [stub.task_by_id[str(task_id)]]

    stub.agents["eng-3"] = {
        "agent_id": "eng-3",
        "agent_type": "engineer",
        "status": "running",
    }

    artifacts = await derive_agent_conversation(
        stub, uuid4(), "eval-task", gateway_task_id=task_id
    )

    assert artifacts.answer_text is None
    assert artifacts.log is None
    assert artifacts.engineer_status is None


@pytest.mark.asyncio
async def test_selects_concise_line_ignoring_trailing_paragraph() -> None:
    stub = _StubEvalClient()
    task_id = uuid4()
    stub.task_by_id[str(task_id)] = {
        "task_id": str(task_id),
        "assigned_agent_ids": ["eng-4"],
        "manager_id": "mgr-1",
    }
    stub.task_rows = [stub.task_by_id[str(task_id)]]

    stub.agents["eng-4"] = {
        "agent_id": "eng-4",
        "agent_type": "engineer",
        "status": "completed",
        "completed_at": "2025-12-03T14:00:00Z",
    }

    stub.messages["eng-4"] = [
        {
            "agent_id": "eng-4",
            "turn_number": 0,
            "role": "assistant",
            "content": {
                "content": [
                    {
                        "block_type": "TextBlock",
                        "data": {"text": "HELLO_AGENT_CLUSTER"},
                    },
                    {
                        "block_type": "TextBlock",
                        "data": {
                            "text": "Please provide me with another task so I can continue working efficiently and share results."
                        },
                    },
                ]
            },
        }
    ]

    artifacts = await derive_agent_conversation(
        stub, uuid4(), "eval-task", gateway_task_id=task_id
    )

    assert artifacts.answer_text == "HELLO_AGENT_CLUSTER"
