from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_ROOT = Path(__file__).resolve().parent / "ai_context"


@lru_cache(maxsize=None)
def load_context_text(*parts: str) -> str:
    path = _ROOT.joinpath(*parts)
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def build_workspace_prompt(workspace: str, task_rules: str) -> str:
    sections = [
        load_context_text("_core", "CONVENTIONS.md"),
        load_context_text(workspace, "CLAUDE.md"),
        task_rules.strip(),
    ]
    return "\n\n".join(section for section in sections if section).strip()

