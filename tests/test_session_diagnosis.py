from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

from core.session import create_session_paths, load_current_session_manifest, summarize_session_report, write_session_manifest


def _iso(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _log_ts(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def test_session_manifest_and_summary_are_session_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    paths = create_session_paths(session_root="sessions", mode="paper", symbol="AVAX/USDT:USDT", timeframe="1m")
    manifest = write_session_manifest(
        paths,
        {
            "created_at": _iso(30),
            "config_path": "config.paper.test.yaml",
            "execution_mode": "paper",
            "symbol": "AVAX/USDT:USDT",
            "timeframe": "1m",
            "resolved_symbol": "AVAX/USDT:USDT",
            "paper_mode": True,
        },
    )

    log_path = tmp_path / manifest["log_file"]
    trade_path = tmp_path / manifest["trade_log_file"]

    log_path.write_text(
        "\n".join(
            [
                f"{_log_ts(20)} [INFO] main: ANALYSIS: Gate: Old gate",
                f"{_log_ts(4)} [INFO] main: ANALYSIS: Signal: HOLD conf=42.0% Reason: recent signal",
                f"{_log_ts(3)} [INFO] main: ANALYSIS: Gate: Range Veto: SELL in bottom 75% of 199c range (pos=14% score=-0.70 sr=0.0 sup_broken=False). Need score>=0.90 + unanimous MTF + sr<0.5 + support_broken.",
            ]
        ),
        encoding="utf-8",
    )

    with trade_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "trade_id", "event", "side", "price", "amount", "pnl", "fees", "score", "confidence", "reason", "signal_reason", "entry_mode", "type"])
        writer.writerow([_iso(4), 1, "ENTRY", "BUY", "9.82", "1.0", "0.00", "0.0000", "0.0", "0.0", "recent entry", "", "RANGE", ""])
        writer.writerow([_iso(20), 2, "EXIT", "SELL", "9.70", "1.0", "-0.20", "0.0000", "0.0", "0.0", "old exit", "", "TREND", ""])

    loaded = load_current_session_manifest("sessions")
    assert loaded is not None
    assert loaded["session_id"] == manifest["session_id"]

    report = summarize_session_report(loaded, lookback_minutes=15)
    assert manifest["session_id"] in report
    assert "recent signal" in report
    assert "Range Veto" in report
    assert "recent entry" in report
    assert "old exit" not in report


def test_session_report_modes_are_selectable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    paths = create_session_paths(session_root="sessions", mode="paper", symbol="AVAX/USDT:USDT", timeframe="3m")
    manifest = write_session_manifest(
        paths,
        {
            "created_at": _iso(10),
            "config_path": "config.paper.test.yaml",
            "execution_mode": "paper",
            "symbol": "AVAX/USDT:USDT",
            "timeframe": "3m",
            "resolved_symbol": "AVAX/USDT:USDT",
            "paper_mode": True,
        },
    )

    log_path = tmp_path / manifest["log_file"]
    trade_path = tmp_path / manifest["trade_log_file"]

    log_path.write_text(
        "\n".join(
            [
                f"{_log_ts(6)} [INFO] main: ANALYSIS: Signal: BUY conf=63.0% Reason: local bounce",
                f"{_log_ts(5)} [INFO] main: ANALYSIS: Signal: HOLD conf=11.0% Reason: midrange veto",
                f"{_log_ts(4)} [INFO] main: ANALYSIS: Gate: SR wall lock: blocked BUY near resistance",
            ]
        ),
        encoding="utf-8",
    )

    with trade_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "trade_id", "event", "side", "price", "amount", "pnl", "fees", "score", "confidence", "reason", "signal_reason", "entry_mode", "type"])
        writer.writerow([_iso(6), 1, "ENTRY", "BUY", "9.81", "1.0", "0.10", "0.0000", "0.0", "0.0", "local bounce", "", "RANGE", ""])

    loaded = load_current_session_manifest("sessions")
    assert loaded is not None

    compact = summarize_session_report(loaded, lookback_minutes=15, report_mode="compact")
    misses = summarize_session_report(loaded, lookback_minutes=15, report_mode="misses")

    assert "Mode: compact" in compact
    assert "Signal actions: BUY=1, HOLD=1" in compact
    assert "Potential misses: HOLD signals=1" in compact
    assert "Recent trades:" in compact

    assert "Mode: misses" in misses
    assert "Recent HOLD signals:" in misses
    assert "Top gates:" in misses
