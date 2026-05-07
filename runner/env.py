"""
Tiny .env loader for ST_Bench. Kept dependency-free so every script that needs
an API key can `from runner.env import load_env_file; load_env_file()` without
requiring python-dotenv.

Priority:
  1. Existing os.environ values (real env always wins — never overwrite).
  2. <project_root>/.env
  3. ~/.stbench_env  (optional fallback)

Supports simple KEY=VALUE lines, with optional surrounding quotes and # comments.
Does not support multiline values or shell expansion.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATE_PATHS = [
    _PROJECT_ROOT / ".env",
    Path.home() / ".stbench_env",
]


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    # Strip surrounding quotes if matched
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_env_file(paths: list[Path] | None = None) -> list[str]:
    """Load KEY=VALUE pairs from the first existing path, adding to os.environ
    only where the key isn't already set. Returns the list of keys added.
    """
    added: list[str] = []
    for p in (paths or _CANDIDATE_PATHS):
        if not p.exists():
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        for raw in text.splitlines():
            kv = _parse_line(raw)
            if kv is None:
                continue
            k, v = kv
            if k not in os.environ and v:
                os.environ[k] = v
                added.append(k)
        break  # first existing file wins
    return added
