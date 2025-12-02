"""Supabase client helpers for the eval harness."""

from __future__ import annotations

import os
from functools import lru_cache

from supabase import Client, create_client


class SupabaseConfigError(RuntimeError):
    """Raised when the Supabase client cannot be configured."""


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return a cached Supabase client instance."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SupabaseConfigError("SUPABASE_URL and SUPABASE_KEY must be set")

    return create_client(url, key)


def reset_supabase_client_cache() -> None:
    """Clear the cached Supabase client (useful for tests)."""
    get_supabase_client.cache_clear()


__all__ = [
    "get_supabase_client",
    "reset_supabase_client_cache",
    "SupabaseConfigError",
]

