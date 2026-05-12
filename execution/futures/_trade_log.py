from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime

from ..base import (
    TRADE_LOG_FILE,
    TRADE_LOG_HEADER,
    _next_trade_id_from_log,
)

logger = logging.getLogger(__name__)

class TradeLogMixin:
    def _record_closed_trade(self, t_type: str, entry: float, exit_price: float, pnl: float, pnl_pct: float, fees: float, side: str = "", reason: str = ""):
        net_pnl = float(pnl) - float(fees)
        trade_record = {
            "type": t_type,
            "entry": float(entry),
            "exit": float(exit_price),
            "pnl": float(net_pnl),
            "pnl_pct": float(pnl_pct),
            "fees": float(fees),
            "side": str(side).upper(),
            "reason": str(reason),
        }
        self.closed_trades.append(trade_record)
        self.stats_trades += 1
        if net_pnl > 0:
            self.stats_wins += 1
        else:
            self.stats_losses += 1
        self.stats_gross += float(pnl)
        self.stats_fees += float(fees)



    def get_session_trades(self) -> list[dict]:
        """Return completed trades with side/reason for session-based ML learning."""
        entry_info = getattr(self, "_entry_info", {})
        trades = []
        for t in self.closed_trades:
            trade_id = int(t.get("trade_id", 0) or 0)
            info = entry_info.get(trade_id, {})
            trades.append({
                "side": t.get("side", "") or info.get("side", ""),
                "reason": t.get("reason", "") or info.get("reason", ""),
                "net_pnl": float(t.get("pnl", 0.0)),
                "entry_price": float(t.get("entry", 0.0)),
                "exit_price": float(t.get("exit", 0.0)),
                "profitable": 1 if float(t.get("pnl", 0.0)) > 0 else 0,
            })
        return trades



    def observe_signal_cycle(self, signal: dict):
        """Remember when the market gives an opposite reset after a profitable exit."""
        try:
            last_prof_side = str(getattr(self, "_last_profitable_exit_side", "") or "").upper()
            if last_prof_side not in {"LONG", "SHORT"}:
                return

            action = str((signal or {}).get("action", "") or "").upper()
            bias = str((signal or {}).get("market_bias", "") or "").upper()
            if last_prof_side == "LONG" and (action == "SELL" or bias.startswith("SHORT")):
                self._opposite_reset_seen_after_profit = True
            elif last_prof_side == "SHORT" and (action == "BUY" or bias.startswith("LONG")):
                self._opposite_reset_seen_after_profit = True
        except Exception:
            pass



    def _same_side_reentry_veto(self, signal: dict, action: str, now: float) -> str:
        """Block repeat entries after a profitable exit unless the cycle reset or signal is very strong."""
        last_prof_side = str(getattr(self, "_last_profitable_exit_side", "") or "").upper()
        last_prof_ts = float(getattr(self, "_last_profitable_exit_ts", 0.0) or 0.0)
        if last_prof_side not in {"LONG", "SHORT"} or last_prof_ts <= 0:
            return ""

        action = str(action or "").upper()
        same_side = (last_prof_side == "LONG" and action == "BUY") or (last_prof_side == "SHORT" and action == "SELL")
        if not same_side:
            return ""

        cooldown = float(getattr(self, "same_side_reentry_cooldown_seconds", 0) or 0)
        elapsed = now - last_prof_ts
        if cooldown > 0 and elapsed < cooldown:
            wait_s = int(max(1.0, cooldown - elapsed))
            return f"Veto: post-profit same-side cooldown ({wait_s}s)"

        if bool(getattr(self, "_opposite_reset_seen_after_profit", False)):
            return ""

        confidence = float((signal or {}).get("confidence", 0.0) or 0.0)
        strong_conf = float(getattr(self, "same_side_reentry_strong_confidence", 0.85) or 0.85)
        if confidence < strong_conf:
            return f"Veto: waiting opposite reset or strong same-side ({confidence:.0%} < {strong_conf:.0%})"
        return ""



    def _init_trade_log(self):
        log_file = getattr(self, "trade_log_file", TRADE_LOG_FILE)
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(TRADE_LOG_HEADER)
        self._next_trade_id = _next_trade_id_from_log(
            log_file,
            int(getattr(self, "_next_trade_id", 1) or 1),
        )



    def _log_trade(
        self,
        trade_id: int,
        event: str,
        side: str,
        price: float,
        amount: float,
        pnl: float = 0.0,
        fees: float = 0.0,
        score: float = 0.0,
        confidence: float = 0.0,
        reason: str = "",
        signal_reason: str = "",
        entry_mode: str = "",
        t_type: str = "",
    ):
        if event.upper() == "ENTRY":
            if not hasattr(self, "_entry_info"):
                self._entry_info = {}
            self._entry_info[int(trade_id)] = {"side": str(side).upper(), "reason": str(reason or ""), "signal_reason": str(signal_reason or ""), "entry_mode": str(entry_mode or "")}
        try:
            log_file = getattr(self, "trade_log_file", TRADE_LOG_FILE)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    int(trade_id),
                    event,
                    side,
                    f"{price:.2f}",
                    f"{amount:.8f}",
                    f"{pnl:.2f}",
                    f"{fees:.4f}",
                    f"{float(score):.6f}",
                    f"{float(confidence):.6f}",
                    (reason or ""),
                    (signal_reason or ""),
                    (entry_mode or ""),
                    (t_type or ""),
                ])
        except Exception as e:
            logger.error(f"Trade Log Error: {e}")

