"""Observability helpers for the eval harness with optional Laminar integration.

When LMNR_PROJECT_API_KEY is set, uses Laminar for real tracing.
Otherwise, falls back to no-op spans for test environments.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

# Try to import Laminar
Laminar = None
observe = None
try:
    from lmnr import Laminar as _Laminar, observe as _observe
    Laminar = _Laminar
    observe = _observe
except ImportError:
    pass

_LAMINAR_INITIALIZED = False


def _ensure_laminar_initialized() -> bool:
    """Initialize Laminar if API key is set and not already initialized.

    Returns:
        True if Laminar is ready to use, False otherwise.
    """
    global _LAMINAR_INITIALIZED

    if _LAMINAR_INITIALIZED:
        return True

    api_key = os.environ.get("LMNR_PROJECT_API_KEY")
    if not api_key:
        return False

    if Laminar is None:
        logger.debug("Laminar SDK not installed, tracing disabled")
        return False

    try:
        Laminar.initialize(project_api_key=api_key)
        _LAMINAR_INITIALIZED = True
        logger.info("Laminar tracing initialized")
        return True
    except Exception as exc:
        logger.warning("Failed to initialize Laminar: %s", exc)
        return False


@contextlib.contextmanager
def traced_span(
    name: str,
    *,
    input: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for creating traced spans.

    When Laminar is available and initialized, creates a real span.
    Otherwise, yields a dict that can be used to record span data (no-op).

    Args:
        name: Span name
        input: Optional input data to record on the span
        metadata: Optional metadata to record on the span

    Yields:
        A dict that can be used to record additional span data.
        In Laminar mode, this is populated with span info.
    """
    span_data: dict[str, Any] = {
        "name": name,
        "input": input,
        "metadata": metadata,
    }

    if _ensure_laminar_initialized() and Laminar is not None:
        try:
            # Use Laminar.start_as_current_span for manual span management
            with Laminar.start_as_current_span(name) as span:
                if input:
                    Laminar.set_span_attributes({"input": str(input)})
                if metadata:
                    Laminar.set_span_attributes(metadata)
                span_data["span"] = span
                yield span_data
        except Exception as exc:
            logger.debug("Laminar span error: %s", exc)
            yield span_data
    else:
        yield span_data


F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator to trace a function with Laminar.

    When Laminar is available, uses @observe for automatic tracing.
    Otherwise, returns the function unchanged.

    Args:
        name: Optional span name (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        if _ensure_laminar_initialized() and observe is not None:
            return observe(name=name or func.__name__)(func)  # type: ignore[return-value]
        return func

    return decorator


def set_span_attributes(attributes: dict[str, Any]) -> None:
    """Set attributes on the current span.

    No-op if Laminar is not initialized.

    Args:
        attributes: Key-value pairs to set on the span
    """
    if _ensure_laminar_initialized() and Laminar is not None:
        try:
            Laminar.set_span_attributes(attributes)
        except Exception as exc:
            logger.debug("Failed to set span attributes: %s", exc)


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled and initialized."""
    return _ensure_laminar_initialized()


__all__ = [
    "traced_span",
    "traced",
    "set_span_attributes",
    "is_tracing_enabled",
]
