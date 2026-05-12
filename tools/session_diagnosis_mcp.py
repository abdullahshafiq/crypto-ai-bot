from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from core.session import load_current_session_manifest, summarize_session_report


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


DEFAULT_SESSION_ROOT = os.getenv("SESSION_ROOT", "sessions")
DEFAULT_LOOKBACK_MINUTES = _env_int("SESSION_DIAG_LOOKBACK_MINUTES", 15)
DEFAULT_MAX_ITEMS = _env_int("SESSION_DIAG_MAX_ITEMS", 8)
DEFAULT_REPORT_MODE = os.getenv("SESSION_DIAG_REPORT_MODE", "compact").strip().lower()
if DEFAULT_REPORT_MODE not in {"compact", "decisions", "misses", "full"}:
    DEFAULT_REPORT_MODE = "compact"


mcp = FastMCP(
    name="crypto-ai-bot-session",
    instructions="Return a compact diagnosis for the current bot session only.",
)


@mcp.tool(
    description=(
        "Summarize the current session from the session-local bot log and trade log. "
        "Choose report_mode from compact, decisions, misses, or full."
    )
)
def current_session_diagnosis(
    lookback_minutes: int | None = None,
    report_mode: Literal["compact", "decisions", "misses", "full"] | None = None,
    max_items: int | None = None,
    session_root: str | None = None,
) -> str:
    lookback_minutes = DEFAULT_LOOKBACK_MINUTES if lookback_minutes is None else max(1, lookback_minutes)
    max_items = DEFAULT_MAX_ITEMS if max_items is None else max(1, max_items)
    report_mode = (report_mode or DEFAULT_REPORT_MODE or "compact").strip().lower()
    if report_mode not in {"compact", "decisions", "misses", "full"}:
        report_mode = "compact"
    session_root = session_root or DEFAULT_SESSION_ROOT

    manifest = load_current_session_manifest(session_root=session_root)
    if not manifest:
        return f"No current session manifest found under {session_root!r}."
    return summarize_session_report(
        manifest,
        lookback_minutes=lookback_minutes,
        max_items=max_items,
        report_mode=report_mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto AI Bot session diagnosis MCP server")
    parser.add_argument("--session-root", default=None, help="Root folder that contains sessions/current/session.json")
    parser.add_argument(
        "--default-lookback-minutes",
        type=int,
        default=None,
        help="Default lookback window used when the tool caller does not override it",
    )
    parser.add_argument(
        "--default-max-items",
        type=int,
        default=None,
        help="Default max items used for recent signals / trades / gates",
    )
    parser.add_argument(
        "--default-report-mode",
        choices=["compact", "decisions", "misses", "full"],
        default=None,
        help="Default report mode used when the tool caller does not override it",
    )
    return parser


def main() -> None:
    global DEFAULT_SESSION_ROOT, DEFAULT_LOOKBACK_MINUTES, DEFAULT_MAX_ITEMS, DEFAULT_REPORT_MODE

    args = _build_parser().parse_args()
    if args.session_root:
        DEFAULT_SESSION_ROOT = args.session_root
    if args.default_lookback_minutes is not None:
        DEFAULT_LOOKBACK_MINUTES = max(1, args.default_lookback_minutes)
    if args.default_max_items is not None:
        DEFAULT_MAX_ITEMS = max(1, args.default_max_items)
    if args.default_report_mode:
        DEFAULT_REPORT_MODE = args.default_report_mode

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
