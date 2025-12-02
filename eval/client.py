"""HTTP client for the Task Gateway REST API."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from uuid import UUID

import httpx

from eval.models import EvalStatus
from eval.observability import traced_span


class GatewayClientError(Exception):
    """Base exception for gateway client errors."""


class GatewayTimeoutError(GatewayClientError):
    """Task polling timed out."""


class GatewayAPIError(GatewayClientError):
    """Gateway API returned an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GatewayClient:
    """Async HTTP client for the Task Gateway API.

    Handles task submission and polling with configurable timeouts
    and retry behavior.
    """

    TERMINAL_STATUSES = frozenset({"completed", "failed"})

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 2.0,
        task_timeout_seconds: float = 300.0,
    ):
        self.base_url = (base_url or os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.task_timeout_seconds = task_timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GatewayClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("GatewayClient must be used as an async context manager")
        return self._client

    async def submit_task(
        self,
        title: str,
        description: str,
        *,
        source: str = "eval",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a new task to the gateway.

        Args:
            title: Task title (eval task ID or short description)
            description: Full prompt or task instructions
            source: Task source identifier (defaults to 'eval')
            metadata: Additional metadata including eval_suite, eval_task_id, etc.

        Returns:
            Task data from the gateway including the assigned task ID

        Raises:
            GatewayAPIError: If the gateway returns an error response
        """
        with traced_span("eval.client.submit_task", input={"title": title}):
            payload = {
                "title": title,
                "description": description,
                "source": "manual",
                "external_source": source,
                "metadata": metadata or {},
            }

            try:
                response = await self.client.post("/api/v1/tasks", json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise GatewayAPIError(
                    f"Failed to submit task: {exc.response.text}",
                    status_code=exc.response.status_code,
                )
            except httpx.RequestError as exc:
                raise GatewayClientError(f"Request failed: {exc}")

    async def get_task(self, task_id: UUID | str) -> dict[str, Any]:
        """Retrieve a task by ID.

        Args:
            task_id: The task UUID

        Returns:
            Task data from the gateway

        Raises:
            GatewayAPIError: If the gateway returns an error response
        """
        with traced_span("eval.client.get_task", input={"task_id": str(task_id)}):
            try:
                response = await self.client.get(f"/api/v1/tasks/{task_id}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise GatewayAPIError(
                    f"Failed to get task {task_id}: {exc.response.text}",
                    status_code=exc.response.status_code,
                )
            except httpx.RequestError as exc:
                raise GatewayClientError(f"Request failed: {exc}")

    async def poll_task_completion(
        self,
        task_id: UUID | str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll a task until it reaches a terminal status.

        Args:
            task_id: The task UUID
            timeout_seconds: Maximum time to wait (defaults to instance setting)
            poll_interval_seconds: Interval between polls (defaults to instance setting)

        Returns:
            Final task data including status and result_metadata

        Raises:
            GatewayTimeoutError: If the task doesn't complete within timeout
            GatewayAPIError: If the gateway returns an error response
        """
        timeout = timeout_seconds or self.task_timeout_seconds
        interval = poll_interval_seconds or self.poll_interval_seconds

        with traced_span(
            "eval.client.poll_task",
            input={"task_id": str(task_id), "timeout": timeout},
        ):
            elapsed = 0.0
            while elapsed < timeout:
                task_data = await self.get_task(task_id)
                status = task_data.get("status", "")

                if status in self.TERMINAL_STATUSES:
                    return task_data

                await asyncio.sleep(interval)
                elapsed += interval

            raise GatewayTimeoutError(
                f"Task {task_id} did not complete within {timeout}s"
            )

    async def submit_and_wait(
        self,
        title: str,
        description: str,
        *,
        source: str = "eval",
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Submit a task and wait for completion.

        Convenience method combining submit_task and poll_task_completion.

        Args:
            title: Task title
            description: Task description/prompt
            source: Task source identifier
            metadata: Additional metadata
            timeout_seconds: Maximum time to wait for completion

        Returns:
            Final task data including status and result_metadata

        Raises:
            GatewayTimeoutError: If the task doesn't complete within timeout
            GatewayAPIError: If the gateway returns an error response
        """
        task = await self.submit_task(title, description, source=source, metadata=metadata)
        task_id = task.get("id")
        if not task_id:
            raise GatewayAPIError("Gateway returned task without ID")

        return await self.poll_task_completion(task_id, timeout_seconds=timeout_seconds)

    async def health_check(self) -> bool:
        """Check gateway health.

        Returns:
            True if gateway is healthy, False otherwise
        """
        try:
            response = await self.client.get("/api/v1/health")
            response.raise_for_status()
            data = response.json()
            return data.get("status") in ("ok", "ready")
        except Exception:
            return False


__all__ = [
    "GatewayClient",
    "GatewayClientError",
    "GatewayTimeoutError",
    "GatewayAPIError",
]
