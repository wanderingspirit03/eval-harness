"""Environment loading helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional
    def load_dotenv(*args, **kwargs):
        pass


@lru_cache(maxsize=1)
def load_project_env() -> None:
    """Load the repo's `.env` file once per process."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)


__all__ = ["load_project_env"]

