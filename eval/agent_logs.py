"""Helpers for hydrating agent logs and deriving final answers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from eval.observability import traced_span
from eval.supabase_client import EvalSupabaseClient


_TERMINAL_AGENT_STATUSES = {"completed", "failed", "killed"}


@dataclass
class AgentConversationArtifacts:
    answer_text: str | None = None
    log: list[dict[str, Any]] | None = None
    engineer_id: str | None = None
    engineer_status: str | None = None
    engineer_error: str | None = None


async def derive_agent_conversation(
    client: EvalSupabaseClient,
    eval_run_id: UUID,
    eval_task_id: str,
    gateway_task_id: UUID | None = None,
) -> AgentConversationArtifacts:
    loop = asyncio.get_running_loop()

    with traced_span(
        "eval.agent_logs.derive_conversation",
        input={"run_id": str(eval_run_id), "eval_task_id": eval_task_id},
    ):
        task_row = None
        if gateway_task_id is not None:
            task_row = await loop.run_in_executor(
                None,
                client.get_task_by_id,
                str(gateway_task_id),
            )

        if task_row is None:
            tasks = await loop.run_in_executor(
                None,
                client.get_tasks_for_eval,
                eval_run_id,
                eval_task_id,
            )
            if tasks:
                task_row = tasks[0]

        if task_row is None:
            return AgentConversationArtifacts()

        assigned_agent_ids = task_row.get("assigned_agent_ids") or []
        manager_id = task_row.get("manager_id")

        engineer_rows = await _fetch_engineers(loop, client, assigned_agent_ids, manager_id)
        terminal_engineers = _filter_terminal_engineers(engineer_rows)
        if not terminal_engineers:
            return AgentConversationArtifacts()

        selected = _select_latest_engineer(terminal_engineers)
        agent_id = selected.get("agent_id")
        if agent_id is None:
            return AgentConversationArtifacts()

        messages = await loop.run_in_executor(None, client.get_messages_for_agent, agent_id)
        normalized_log = _normalize_messages(messages)
        answer_text = _derive_answer_text(normalized_log, selected.get("result"))

        return AgentConversationArtifacts(
            answer_text=answer_text,
            log=normalized_log or None,
            engineer_id=agent_id,
            engineer_status=selected.get("status"),
            engineer_error=selected.get("error"),
        )


async def derive_agent_answer_and_log(
    client: EvalSupabaseClient,
    eval_run_id: UUID,
    eval_task_id: str,
    gateway_task_id: UUID | None = None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    artifacts = await derive_agent_conversation(
        client,
        eval_run_id,
        eval_task_id,
        gateway_task_id=gateway_task_id,
    )
    return artifacts.answer_text, artifacts.log


async def _fetch_engineers(
    loop: asyncio.AbstractEventLoop,
    client: EvalSupabaseClient,
    assigned_ids: list[str],
    manager_id: str | None,
) -> list[dict[str, Any]]:
    if assigned_ids:
        rows = await loop.run_in_executor(None, client.get_agents_by_ids, assigned_ids)
        if rows:
            return rows

    if manager_id is None:
        return []

    rows = await loop.run_in_executor(None, client.get_engineers_for_manager, manager_id)
    return rows


def _filter_terminal_engineers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal: list[dict[str, Any]] = []
    for row in rows:
        if row.get("agent_type") != "engineer":
            continue
        status = row.get("status")
        if status not in _TERMINAL_AGENT_STATUSES:
            continue
        terminal.append(row)
    return terminal


def _select_latest_engineer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _sort_key(entry: dict[str, Any]) -> tuple[datetime, str]:
        timestamp = _parse_datetime(entry.get("completed_at"))
        if timestamp is None:
            timestamp = _parse_datetime(entry.get("updated_at"))
        if timestamp is None:
            timestamp = datetime.fromtimestamp(0, tz=timezone.utc)
        agent_id = entry.get("agent_id")
        if isinstance(agent_id, str):
            return (timestamp, agent_id)
        return (timestamp, "")

    sorted_rows = sorted(rows, key=_sort_key)
    return sorted_rows[-1]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in sorted(messages, key=lambda item: item.get("turn_number", 0)):
        role = raw.get("role")
        text = _render_message_text(raw.get("content"))
        if not text:
            continue
        normalized.append(
            {
                "role": role,
                "text": text,
                "turn_number": raw.get("turn_number"),
                "timestamp": raw.get("timestamp"),
            }
        )
    return normalized


def _render_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        blocks = content.get("content")
        texts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                text_block = _extract_text_from_block(block)
                if text_block:
                    texts.append(text_block)
        result_text = content.get("result")
        if isinstance(result_text, str):
            texts.append(result_text)
        combined = "\n".join(part.strip() for part in texts if part and part.strip())
        return combined.strip()
    return ""


def _extract_text_from_block(block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    data = block.get("data")
    if isinstance(data, dict):
        text_value = data.get("text")
        if isinstance(text_value, str):
            return text_value
        content_value = data.get("content")
        if isinstance(content_value, str):
            return content_value
    return ""


def _derive_answer_text(log: list[dict[str, Any]], agent_result: Any) -> str | None:
    for entry in reversed(log):
        role = entry.get("role")
        text = entry.get("text")
        if role == "assistant" and isinstance(text, str) and text.strip():
            candidate = _select_preferred_line(text)
            if candidate is not None:
                return candidate

    if isinstance(agent_result, dict):
        for key in ("answer_text", "result", "summary"):
            value = agent_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _last_non_empty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines()]
    for line in reversed(lines):
        if line:
            return line
    stripped = text.strip()
    if stripped:
        return stripped
    return None


def _select_preferred_line(text: str) -> str | None:
    """Select the most likely final answer line from a text block."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Highest priority: explicit "final answer" markers (case-insensitive)
    for line in reversed(lines):
        lowered = line.lower()
        if "final answer" in lowered or "answer:" in lowered:
            # If the line is "Final Answer: XYZ", return "XYZ"
            if ":" in line:
                suffix = line.split(":", 1)[1].strip()
                if suffix:
                    return suffix
            # Otherwise return the whole line if it seems substantial
            return line

    # Next priority: Look for specific format indicators used by engineers
    # e.g. XML tags like <answer>...</answer> or JSON blocks
    for line in reversed(lines):
        if line.startswith("<answer>") and line.endswith("</answer>"):
            return line[8:-9].strip()
        if line.startswith("{") and line.endswith("}") and "answer" in line:
            # Simple JSON heuristic
            return line

    # Priority: Short, declarative lines (≤20 words) at the end often contain the answer
    # Relaxed from 12 to 20 to capture slightly longer sentences
    for line in reversed(lines[:5]): # Check last 5 lines
        if len(line.split()) <= 20:
            return line

    # Fallback: The very last non-empty line
    return lines[-1]


__all__ = [
    "AgentConversationArtifacts",
    "derive_agent_answer_and_log",
    "derive_agent_conversation",
]
