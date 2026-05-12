from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    session_root: str
    session_dir: str
    log_file: str
    trade_log_file: str
    manifest_file: str
    current_manifest_file: str
    created_at: str


def _slugify(value: Any, fallback: str = "session") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or fallback


def build_session_id(mode: str = "paper", symbol: str = "", timeframe: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [_slugify(mode, "paper"), stamp]
    sym = _slugify(symbol, "")
    tf = _slugify(timeframe, "")
    if sym:
        parts.append(sym)
    if tf:
        parts.append(tf)
    return "_".join(parts)


def create_session_paths(
    session_root: str = "sessions",
    mode: str = "paper",
    symbol: str = "",
    timeframe: str = "",
    trade_log_name: str = "trade_log.csv",
) -> SessionPaths:
    root = Path(session_root)
    session_id = build_session_id(mode=mode, symbol=symbol, timeframe=timeframe)
    session_dir = root / session_id
    current_dir = root / "current"
    session_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return SessionPaths(
        session_id=session_id,
        session_root=str(root),
        session_dir=str(session_dir),
        log_file=str(session_dir / "bot.log"),
        trade_log_file=str(session_dir / trade_log_name),
        manifest_file=str(session_dir / "session.json"),
        current_manifest_file=str(current_dir / "session.json"),
        created_at=created_at,
    )


def write_session_manifest(paths: SessionPaths, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": paths.session_id,
        "session_root": paths.session_root,
        "session_dir": paths.session_dir,
        "log_file": paths.log_file,
        "trade_log_file": paths.trade_log_file,
        "created_at": paths.created_at,
    }
    if metadata:
        payload.update(metadata)

    for file_name in (paths.manifest_file, paths.current_manifest_file):
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_current_session_manifest(session_root: str = "sessions") -> dict[str, Any] | None:
    current_file = Path(session_root) / "current" / "session.json"
    if not current_file.exists():
        return None
    try:
        return json.loads(current_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_log_timestamp(line: str) -> datetime | None:
    match = _LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _recent_lines_by_time(file_path: str, since: datetime) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        return []
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ts = _parse_log_timestamp(raw_line)
        if ts is None or ts < since:
            continue
        lines.append(raw_line.strip())
    return lines


def _recent_trade_rows(file_path: str, since: datetime) -> list[dict[str, str]]:
    path = Path(file_path)
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_raw = str(row.get("timestamp", "") or "").strip()
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                continue
            if ts >= since:
                rows.append(row)
    return rows


def _signal_action_from_line(line: str) -> str | None:
    match = re.search(r"Signal:\s*(BUY|SELL|HOLD)\b", line, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def summarize_session_report(
    manifest: dict[str, Any],
    lookback_minutes: int = 15,
    max_items: int = 8,
    report_mode: str = "compact",
) -> str:
    if not manifest:
        return "No active session manifest found."

    report_mode = (report_mode or "compact").strip().lower()
    if report_mode not in {"compact", "decisions", "misses", "full"}:
        report_mode = "compact"

    session_id = str(manifest.get("session_id", "") or "unknown")
    session_dir = str(manifest.get("session_dir", "") or "")
    log_file = str(manifest.get("log_file", "") or "")
    trade_log_file = str(manifest.get("trade_log_file", "") or "")
    created_at_raw = str(manifest.get("created_at", "") or "")
    try:
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    except Exception:
        created_at = datetime.now(timezone.utc) - timedelta(minutes=max(lookback_minutes, 1))

    window_start = max(created_at, datetime.now(timezone.utc) - timedelta(minutes=max(lookback_minutes, 1)))
    log_lines = _recent_lines_by_time(log_file, window_start)
    trade_rows = _recent_trade_rows(trade_log_file, window_start)

    signal_lines = [line for line in log_lines if "ANALYSIS: Signal:" in line]
    gate_lines = [line for line in log_lines if "ANALYSIS: Gate:" in line]
    signal_counter = Counter()
    gate_counter = Counter()
    hold_signal_lines: list[str] = []
    for line in signal_lines:
        action = _signal_action_from_line(line)
        if action:
            signal_counter[action] += 1
            if action == "HOLD":
                hold_signal_lines.append(line)
    for line in gate_lines:
        _, _, tail = line.partition("Gate:")
        gate_counter[tail.strip()] += 1

    event_counter = Counter()
    side_counter = Counter()
    for row in trade_rows:
        event_counter[str(row.get("event", "") or "").upper()] += 1
        side_counter[str(row.get("side", "") or "").upper()] += 1

    report: list[str] = []
    report.append(f"Session: {session_id}")
    report.append(f"Mode: {report_mode}")
    report.append(f"Window: {window_start.isoformat(timespec='seconds')} -> now")
    if session_dir:
        report.append(f"Dir: {session_dir}")
    if log_file:
        report.append(f"Log: {log_file}")
    if trade_log_file:
        report.append(f"Trade log: {trade_log_file}")
    report.append("")
    report.append(f"Signals in window: {len(signal_lines)}")
    report.append(f"Gate lines in window: {len(gate_lines)}")
    if signal_counter:
        report.append(
            "Signal actions: "
            + ", ".join(f"{name}={count}" for name, count in sorted(signal_counter.items()))
        )
    if hold_signal_lines:
        report.append(f"Potential misses: HOLD signals={len(hold_signal_lines)}")
    if event_counter:
        report.append(
            "Trade events: "
            + ", ".join(f"{name}={count}" for name, count in sorted(event_counter.items()))
        )
    if side_counter:
        report.append(
            "Trade sides: "
            + ", ".join(f"{name or 'UNKNOWN'}={count}" for name, count in sorted(side_counter.items()))
        )

    if gate_counter:
        report.append("")
        report.append("Top gates:")
        for reason, count in gate_counter.most_common(max_items):
            report.append(f"- {count}x {reason}")

    if signal_lines and report_mode in {"compact", "decisions", "full"}:
        report.append("")
        report.append("Recent signals:")
        for line in signal_lines[-max_items:]:
            report.append(f"- {line}")

    if hold_signal_lines and report_mode in {"misses", "full"}:
        report.append("")
        report.append("Recent HOLD signals:")
        for line in hold_signal_lines[-max_items:]:
            report.append(f"- {line}")

    if trade_rows and report_mode in {"compact", "decisions", "full"}:
        report.append("")
        report.append("Recent trades:")
        for row in trade_rows[-max_items:]:
            report.append(
                "- {timestamp} {event} {side} {price} pnl={pnl} reason={reason}".format(
                    timestamp=row.get("timestamp", ""),
                    event=row.get("event", ""),
                    side=row.get("side", ""),
                    price=row.get("price", ""),
                    pnl=row.get("pnl", ""),
                    reason=row.get("reason", ""),
                )
            )

    return "\n".join(report)
