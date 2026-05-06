from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from execution import resolve_data_market
from indicators import (
    build_mtf_timeframe_context,
    calculate_base_indicators,
    compute_advanced_pivots,
    generate_quant_signal,
)
from market import MarketData


DEFAULT_TIMEFRAMES = ["3m", "5m", "10m", "15m", "1h", "4h"]


@dataclass
class BacktestTrade:
    trade_id: int
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    pnl_pct: float
    fees: float
    exit_type: str
    bars_held: int
    signal_score: float = 0.0
    signal_confidence: float = 0.0
    signal_reason: str = ""
    entry_mode: str = ""


@dataclass
class BacktestSummary:
    symbol: str
    timeframe: str
    bars_processed: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    ending_balance: float = 0.0
    starting_balance: float = 0.0
    avg_hold_bars: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_by_side: dict[str, int] = field(default_factory=dict)
    trades_by_exit: dict[str, int] = field(default_factory=dict)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _slice_up_to(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["timestamp"] <= cutoff_ts
    return df.loc[mask].copy()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _build_state_from_window(
    df_window: pd.DataFrame,
    latest_indicators: dict[str, Any],
    spread_pct: float,
) -> dict[str, Any]:
    last = df_window.iloc[-1]
    prev = df_window.iloc[-2] if len(df_window) > 1 else last
    day_open = float(df_window.iloc[0]["open"])
    bb_mid = _safe_float(latest_indicators.get("bb_mid"), day_open)
    control_zone = _safe_float(latest_indicators.get("vwap"), bb_mid)
    return {
        "price": float(last["close"]),
        "spread_pct": float(spread_pct),
        "ret_30s": 0.0,
        "ret_5s": 0.0,
        "volume_state": "normal",
        "orderbook": {},
        "session_open": day_open,
        "previous_close": float(prev["close"]),
        "previous_high": float(prev["high"]),
        "previous_low": float(prev["low"]),
        "control_zone": control_zone,
        "average_zone": control_zone,
    }


def _latest_macro_from_context(mtf_context: dict[str, dict[str, Any]], funding_rate: float = 0.0) -> dict[str, Any]:
    h1 = mtf_context.get("1h", {}) if isinstance(mtf_context, dict) else {}
    trend = str(h1.get("trend", "NEUT") or "NEUT").upper()
    if trend == "BULL":
        regime = "BULLISH"
    elif trend == "BEAR":
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"
    return {"regime": regime, "bias": regime, "funding_rate": float(funding_rate or 0.0)}


def _extract_entry_mode(reason: str) -> str:
    if not reason:
        return "TREND"
    marker = "[Mode:"
    if marker not in reason:
        return "TREND"
    try:
        segment = reason.split(marker, 1)[1]
        return segment.split("]", 1)[0] or "TREND"
    except Exception:
        return "TREND"


def _entry_fill_price(next_open: float, side: str, spread_pct: float, slippage_pct: float) -> float:
    side_u = str(side or "").upper()
    base = float(next_open)
    if side_u == "BUY":
        return base * (1.0 + max(0.0, spread_pct / 2.0) + max(0.0, slippage_pct))
    return base * (1.0 - max(0.0, spread_pct / 2.0) - max(0.0, slippage_pct))


def _dynamic_leverage_from_confidence(cfg: dict[str, Any], confidence: float) -> float:
    lev_cfg = cfg.get("leverage", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    base_lev = float(exec_cfg.get("leverage", 1) or 1)
    if not bool(lev_cfg.get("enabled", True)):
        return max(1.0, base_lev)

    min_lev = float(lev_cfg.get("min_leverage", 1) or 1)
    max_lev = float(lev_cfg.get("max_leverage", base_lev) or base_lev)
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    if confidence < 0.20:
        target = min_lev
    elif confidence < 0.35:
        target = min_lev + (max_lev - min_lev) * 0.35
    elif confidence < 0.50:
        target = min_lev + (max_lev - min_lev) * 0.60
    else:
        target = max_lev
    return float(max(min_lev, min(max_lev, target)))


def _prepare_history_bundle(
    market: MarketData,
    symbol: str,
    base_tf: str,
    timeframes: list[str],
    history_limit: int,
) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    requested = [base_tf] + [tf for tf in timeframes if tf != base_tf] + ["1d"]
    for tf in dict.fromkeys(requested):
        limit = history_limit
        if tf == "1d":
            limit = max(60, min(history_limit, 120))
        elif tf.endswith("h"):
            limit = max(120, min(history_limit, 250))
        elif tf.endswith("m"):
            limit = max(240, min(history_limit, 1000))
        df = market.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if df is None or df.empty:
            bundle[tf] = pd.DataFrame()
        else:
            bundle[tf] = df.sort_values("timestamp").reset_index(drop=True)
    return bundle


def _build_mtf_context(
    bundle: dict[str, pd.DataFrame],
    cutoff_ts: pd.Timestamp,
    tfs: list[str],
) -> dict[str, dict[str, Any]]:
    mtf_context: dict[str, dict[str, Any]] = {}
    for tf in tfs:
        df = bundle.get(tf)
        if df is None or df.empty:
            continue
        sliced = _slice_up_to(df, cutoff_ts)
        if len(sliced) < 50:
            continue
        try:
            mtf_context[tf] = build_mtf_timeframe_context(calculate_base_indicators(sliced))
        except Exception:
            continue
    return mtf_context


def _build_pivot_data(bundle: dict[str, pd.DataFrame], cutoff_ts: pd.Timestamp) -> dict[str, Any]:
    daily = bundle.get("1d")
    if daily is None or daily.empty:
        return {}
    sliced = _slice_up_to(daily, cutoff_ts)
    if len(sliced) < 2:
        return {}
    try:
        return compute_advanced_pivots(sliced)
    except Exception:
        return {}


def _close_position(
    position: dict[str, Any],
    exit_price: float,
    exit_time: pd.Timestamp,
    exit_type: str,
    bars_held: int,
    fee_rate: float,
) -> BacktestTrade:
    entry = float(position["entry"])
    amount = float(position["amount"])
    side = str(position["side"])
    if side == "LONG":
        pnl = (exit_price - entry) * amount
        pnl_pct = ((exit_price - entry) / entry) * 100.0
    else:
        pnl = (entry - exit_price) * amount
        pnl_pct = ((entry - exit_price) / entry) * 100.0
    fees = (amount * entry * fee_rate) + (amount * exit_price * fee_rate)
    pnl_after_fees = pnl - fees
    return BacktestTrade(
        trade_id=int(position["trade_id"]),
        side=side,
        entry_time=str(position["entry_time"]),
        exit_time=str(exit_time),
        entry_price=entry,
        exit_price=float(exit_price),
        amount=amount,
        pnl=float(pnl_after_fees),
        pnl_pct=float(pnl_pct),
        fees=float(fees),
        exit_type=str(exit_type),
        bars_held=int(bars_held),
        signal_score=float(position.get("signal_score", 0.0) or 0.0),
        signal_confidence=float(position.get("signal_confidence", 0.0) or 0.0),
        signal_reason=str(position.get("signal_reason", "") or ""),
        entry_mode=str(position.get("entry_mode", "") or ""),
    )


def _process_open_position(
    position: dict[str, Any],
    bar: pd.Series,
    fee_rate: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, BacktestTrade | None]:
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    ts = bar["timestamp"]
    side = str(position["side"])
    entry = float(position["entry"])
    sl = float(position["sl"])
    tp = float(position["tp"])
    highest = float(position.get("highest_price", entry))
    lowest = float(position.get("lowest_price", entry))
    trail_armed = bool(position.get("trail_armed", False))
    break_even_trigger = float(position.get("break_even_trigger_pct", 0.0) or 0.0)
    break_even_buffer = float(position.get("break_even_buffer_pct", 0.0) or 0.0)
    t1_trigger = float(position.get("trail_tighten_1_pct", 0.0) or 0.0)
    t2_trigger = float(position.get("trail_tighten_2_pct", 0.0) or 0.0)
    t1_gap = float(position.get("trail_t1_gap_pct", 0.0) or 0.0)
    t2_gap = float(position.get("trail_t2_gap_pct", 0.0) or 0.0)
    min_hold_seconds = int(position.get("min_hold_seconds", 0) or 0)
    entry_ts = pd.Timestamp(position["entry_time"])
    hold_seconds = max(0.0, (pd.Timestamp(ts) - entry_ts).total_seconds())

    if side == "LONG":
        highest = max(highest, high)
        profit_pct = (close - entry) / entry
        if not trail_armed and profit_pct >= break_even_trigger:
            trail_armed = True
            sl = max(sl, entry * (1.0 + break_even_buffer))
        if trail_armed:
            if profit_pct >= t2_trigger and t2_gap > 0:
                sl = max(sl, highest * (1.0 - t2_gap))
            elif profit_pct >= t1_trigger and t1_gap > 0:
                sl = max(sl, highest * (1.0 - t1_gap))

        tp_hit = high >= tp
        sl_hit = low <= sl
        trail_hit = sl_hit
        if tp_hit and sl_hit:
            trade = _close_position(
                position,
                sl,
                ts,
                "STOP_LOSS",
                position.get("bars_held", 0) + 1,
                fee_rate,
            )
            return None, trade
        if sl_hit:
            exit_type = "TRAIL_SL" if trail_armed else "STOP_LOSS"
            trade = _close_position(
                position,
                sl,
                ts,
                exit_type,
                position.get("bars_held", 0) + 1,
                fee_rate,
            )
            return None, trade
        if tp_hit:
            trade = _close_position(
                position,
                tp,
                ts,
                "TAKE_PROFIT",
                position.get("bars_held", 0) + 1,
                fee_rate,
            )
            return None, trade
        position["highest_price"] = highest
        position["trail_armed"] = trail_armed
        position["sl"] = sl
        position["bars_held"] = position.get("bars_held", 0) + 1
        position["hold_seconds"] = hold_seconds
        return position, None

    # SHORT
    lowest = min(lowest, low)
    profit_pct = (entry - close) / entry
    if not trail_armed and profit_pct >= break_even_trigger:
        trail_armed = True
        sl = min(sl, entry * (1.0 - break_even_buffer))
    if trail_armed:
        if profit_pct >= t2_trigger and t2_gap > 0:
            sl = min(sl, lowest * (1.0 + t2_gap))
        elif profit_pct >= t1_trigger and t1_gap > 0:
            sl = min(sl, lowest * (1.0 + t1_gap))

    tp_hit = low <= tp
    sl_hit = high >= sl
    if tp_hit and sl_hit:
        trade = _close_position(
            position,
            sl,
            ts,
            "STOP_LOSS",
            position.get("bars_held", 0) + 1,
            fee_rate,
        )
        return None, trade
    if sl_hit:
        exit_type = "TRAIL_SL" if trail_armed else "STOP_LOSS"
        trade = _close_position(
            position,
            sl,
            ts,
            exit_type,
            position.get("bars_held", 0) + 1,
            fee_rate,
        )
        return None, trade
    if tp_hit:
        trade = _close_position(
            position,
            tp,
            ts,
            "TAKE_PROFIT",
            position.get("bars_held", 0) + 1,
            fee_rate,
        )
        return None, trade
    position["lowest_price"] = lowest
    position["trail_armed"] = trail_armed
    position["sl"] = sl
    position["bars_held"] = position.get("bars_held", 0) + 1
    position["hold_seconds"] = hold_seconds
    return position, None


class BacktestEngine:
    def __init__(self, cfg: dict[str, Any], symbol: str | None = None, timeframe: str | None = None, limit: int = 1000):
        self.cfg = cfg or {}
        self.symbol = symbol or self.cfg.get("symbol", "AVAX/USDC:USDC")
        self.timeframe = timeframe or self.cfg.get("timeframe", "5m")
        self.base_timeframe = self.timeframe
        self.mtf_timeframes = [tf for tf in DEFAULT_TIMEFRAMES if tf != self.base_timeframe]
        self.history_limit = int(limit or 1000)
        self.market = MarketData(resolve_data_market(self.cfg))
        self.strategy_cfg = dict(self.cfg.get("strategy", {}) or {})
        self.mtf_cfg = dict(self.cfg.get("mtf", {}) or {})
        self.execution_cfg = dict(self.cfg.get("execution", {}) or {})
        self.starting_balance = float(
            self.cfg.get("backtest", {}).get(
                "starting_balance_usdt",
                self.execution_cfg.get("paper_starting_balance_usdt", 1000.0),
            )
            or 1000.0
        )
        self.fixed_trade_usdt = float(self.strategy_cfg.get("fixed_trade_usdt", 25.0) or 25.0)
        self.fee_rate = float(self.execution_cfg.get("fee_rate", 0.0006) or 0.0006)
        self.spread_pct = float(self.cfg.get("backtest", {}).get("spread_pct", 0.00022) or 0.00022)
        self.slippage_pct = float(self.cfg.get("backtest", {}).get("slippage_pct", 0.00015) or 0.00015)
        self.allow_reversal = bool(self.cfg.get("backtest", {}).get("allow_reversal", True))
        self.backtest_funding_rate = float(self.cfg.get("backtest", {}).get("funding_rate", 0.0) or 0.0)
        self.use_live_funding_rate = bool(self.cfg.get("backtest", {}).get("use_live_funding_rate", False))
        self.history_bundle: dict[str, pd.DataFrame] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[dict[str, Any]] = []

    def load_data(self) -> None:
        self.history_bundle = _prepare_history_bundle(
            self.market,
            self.symbol,
            self.base_timeframe,
            self.mtf_timeframes,
            self.history_limit,
        )

    def run(self) -> BacktestSummary:
        if not self.history_bundle:
            self.load_data()

        base_df = self.history_bundle.get(self.base_timeframe, pd.DataFrame())
        if base_df is None or base_df.empty:
            raise RuntimeError(f"No OHLCV data available for {self.symbol} @ {self.base_timeframe}")

        warmup = max(80, 60)
        balance = self.starting_balance
        peak_equity = balance
        max_drawdown = 0.0
        position: dict[str, Any] | None = None
        pending_entry: dict[str, Any] | None = None
        trade_id = 1

        for i in range(warmup, len(base_df) - 1):
            bar = base_df.iloc[i]
            cutoff_ts = pd.Timestamp(bar["timestamp"])
            next_bar = base_df.iloc[i + 1]

            if pending_entry is not None:
                side = pending_entry["side"]
                entry_price = _entry_fill_price(
                    float(next_bar["open"]),
                    side,
                    self.spread_pct,
                    self.slippage_pct,
                )
                amount = (
                    pending_entry["trade_usdt"] * pending_entry["leverage"]
                ) / entry_price
                default_sl = (
                    entry_price * (0.996 if side == "BUY" else 1.004)
                )
                default_tp = (
                    entry_price * (1.0025 if side == "BUY" else 0.9975)
                )
                position = {
                    "trade_id": pending_entry["trade_id"],
                    "side": "LONG" if side == "BUY" else "SHORT",
                    "entry": entry_price,
                    "amount": amount,
                    "entry_time": str(next_bar["timestamp"]),
                    "highest_price": entry_price,
                    "lowest_price": entry_price,
                    "trail_armed": False,
                    "sl": float(
                        pending_entry["signal"].get("sl", default_sl)
                        or entry_price
                    ),
                    "tp": float(
                        pending_entry["signal"].get("tp", default_tp)
                        or entry_price
                    ),
                    "break_even_trigger_pct": float(
                        self.execution_cfg.get("break_even_trigger_pct", 0.0015)
                        or 0.0015
                    ),
                    "break_even_buffer_pct": float(
                        self.execution_cfg.get("break_even_buffer_pct", 0.0004)
                        or 0.0004
                    ),
                    "trail_tighten_1_pct": float(
                        self.execution_cfg.get("trail_tighten_1_pct", 0.0020)
                        or 0.0020
                    ),
                    "trail_tighten_2_pct": float(
                        self.execution_cfg.get("trail_tighten_2_pct", 0.0040)
                        or 0.0040
                    ),
                    "trail_t1_gap_pct": float(
                        self.execution_cfg.get("trail_t1_gap_pct", 0.0025)
                        or 0.0025
                    ),
                    "trail_t2_gap_pct": float(
                        self.execution_cfg.get("trail_t2_gap_pct", 0.0020)
                        or 0.0020
                    ),
                    "min_hold_seconds": int(
                        self.execution_cfg.get("scalp_min_hold_seconds", 10) or 10
                    ),
                    "bars_held": 0,
                    "signal_score": float(
                        pending_entry["signal"].get("score", 0.0) or 0.0
                    ),
                    "signal_confidence": float(
                        pending_entry["signal"].get("confidence", 0.0) or 0.0
                    ),
                    "signal_reason": str(
                        pending_entry["signal"].get("reason", "") or ""
                    ),
                    "entry_mode": str(
                        pending_entry.get("entry_mode", "") or ""
                    ),
                }
                pending_entry = None

            if position is not None:
                position, closed_trade = _process_open_position(
                    position,
                    bar,
                    self.fee_rate,
                    self.execution_cfg,
                )
                if closed_trade is not None:
                    self.trades.append(closed_trade)
                    balance += closed_trade.pnl
                    position = None

            base_window = base_df.iloc[: i + 1].copy()
            if len(base_window) < 60:
                continue
            base_ind = calculate_base_indicators(base_window)
            latest_indicators = base_ind.iloc[-1].to_dict()
            signal_df = base_ind.iloc[:-1] if len(base_ind) > 1 else base_ind
            mtf_context = _build_mtf_context(
                self.history_bundle,
                cutoff_ts,
                self.mtf_timeframes,
            )
            pivots = _build_pivot_data(self.history_bundle, cutoff_ts)
            if self.use_live_funding_rate and i == warmup:
                fr = float(self.market.fetch_funding_rate(self.symbol) or 0.0)
            else:
                fr = self.backtest_funding_rate
            latest_macro = _latest_macro_from_context(mtf_context, funding_rate=fr)
            state = _build_state_from_window(
                base_window,
                latest_indicators,
                self.spread_pct,
            )

            signal = generate_quant_signal(
                state,
                latest_indicators,
                self.strategy_cfg,
                signal_df,
                latest_macro,
                mtf_context=mtf_context,
                mtf_config=self.mtf_cfg,
                pivot_data=pivots,
            )

            if position is not None and self.allow_reversal:
                current_side = position["side"]
                action = signal.get("action", "HOLD")
                is_reversing = (
                    (current_side == "LONG" and action == "SELL")
                    or (current_side == "SHORT" and action == "BUY")
                )
                if is_reversing:
                    entry = float(position["entry"])
                    current_price = float(bar["close"])
                    if current_side == "LONG":
                        profit_pct = (current_price - entry) / entry
                    else:
                        profit_pct = (entry - current_price) / entry
                    net_edge = float(
                        self.execution_cfg.get("reversal_min_net_edge_pct", 0.0020)
                        or 0.0020
                    )
                    min_hold_seconds = float(
                        position.get("min_hold_seconds", 0.0) or 0.0
                    )
                    hold_seconds = float(
                        position.get("hold_seconds", 0.0) or 0.0
                    )
                    if (
                        profit_pct >= net_edge
                        and hold_seconds >= min_hold_seconds
                    ):
                        closed_trade = _close_position(
                            position,
                            current_price,
                            bar["timestamp"],
                            "REVERSAL_BANK",
                            position.get("bars_held", 0),
                            self.fee_rate,
                        )
                        self.trades.append(closed_trade)
                        balance += closed_trade.pnl
                        position = None

            if (
                signal.get("action") in {"BUY", "SELL"}
                and position is None
                and pending_entry is None
            ):
                confidence = float(signal.get("confidence", 0.0) or 0.0)
                score = float(signal.get("score", 0.0) or 0.0)
                min_conf = float(
                    self.strategy_cfg.get("min_conf", 0.15) or 0.15
                )
                min_score = float(
                    self.strategy_cfg.get("entry_min_confidence_hard", 0.20)
                    or 0.20
                )
                if confidence >= min_conf and abs(score) >= min_score:
                    if self.fixed_trade_usdt > 0:
                        trade_usdt = min(self.fixed_trade_usdt, balance * 0.90)
                    else:
                        trade_usdt = balance * 0.25
                    if trade_usdt >= 10.0:
                        leverage = _dynamic_leverage_from_confidence(
                            self.cfg,
                            confidence,
                        )
                        pending_entry = {
                            "trade_id": trade_id,
                            "side": signal["action"],
                            "signal": copy.deepcopy(signal),
                            "trade_usdt": trade_usdt,
                            "leverage": leverage,
                            "entry_mode": _extract_entry_mode(
                                str(signal.get("reason", "") or "")
                            ),
                        }
                        trade_id += 1

            unrealized = 0.0
            if position is not None:
                current_price = float(bar["close"])
                entry = float(position["entry"])
                amount = float(position["amount"])
                if position["side"] == "LONG":
                    unrealized = (current_price - entry) * amount
                else:
                    unrealized = (entry - current_price) * amount
            equity = balance + unrealized
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            self.equity_curve.append(
                {
                    "timestamp": str(bar["timestamp"]),
                    "equity": float(equity),
                    "balance": float(balance),
                    "unrealized": float(unrealized),
                    "position": position["side"] if position else "FLAT",
                }
            )

        if position is not None:
            final_bar = base_df.iloc[-1]
            closed_trade = _close_position(
                position,
                float(final_bar["close"]),
                final_bar["timestamp"],
                "EOD_CLOSE",
                position.get("bars_held", 0),
                self.fee_rate,
            )
            self.trades.append(closed_trade)
            balance += closed_trade.pnl

        summary = self._summarize(balance, max_drawdown)
        return summary

    def _summarize(
        self,
        ending_balance: float,
        max_drawdown: float,
    ) -> BacktestSummary:
        summary = BacktestSummary(
            symbol=self.symbol,
            timeframe=self.base_timeframe,
            bars_processed=len(self.equity_curve),
            starting_balance=float(self.starting_balance),
            ending_balance=float(ending_balance),
            max_drawdown=float(max_drawdown),
        )
        summary.trades = len(self.trades)
        summary.wins = sum(1 for t in self.trades if t.pnl > 0)
        summary.losses = sum(1 for t in self.trades if t.pnl <= 0)
        summary.win_rate = (summary.wins / summary.trades) if summary.trades else 0.0
        summary.gross_profit = sum(
            t.pnl for t in self.trades if t.pnl > 0
        )
        summary.gross_loss = sum(t.pnl for t in self.trades if t.pnl < 0)
        summary.net_pnl = sum(t.pnl for t in self.trades)
        if summary.gross_loss < 0:
            summary.profit_factor = (
                summary.gross_profit / abs(summary.gross_loss)
            )
        else:
            summary.profit_factor = 0.0
        if summary.trades:
            summary.expectancy = summary.net_pnl / summary.trades
        else:
            summary.expectancy = 0.0
        avg_win_trades = [t.pnl for t in self.trades if t.pnl > 0]
        avg_loss_trades = [t.pnl for t in self.trades if t.pnl <= 0]
        summary.avg_win = sum(avg_win_trades) / len(avg_win_trades) if avg_win_trades else 0.0
        summary.avg_loss = sum(avg_loss_trades) / len(avg_loss_trades) if avg_loss_trades else 0.0
        if self.trades:
            summary.avg_hold_bars = (
                sum(t.bars_held for t in self.trades) / len(self.trades)
            )
        else:
            summary.avg_hold_bars = 0.0
        summary.trades_by_side = {
            "LONG": sum(1 for t in self.trades if t.side == "LONG"),
            "SHORT": sum(1 for t in self.trades if t.side == "SHORT"),
        }
        exit_counts: dict[str, int] = {}
        for t in self.trades:
            exit_counts[t.exit_type] = exit_counts.get(t.exit_type, 0) + 1
        summary.trades_by_exit = exit_counts
        return summary

    def export_trades(self, path: str | Path) -> None:
        rows = [asdict(t) for t in self.trades]
        df = pd.DataFrame(rows)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def export_equity(self, path: str | Path) -> None:
        df = pd.DataFrame(self.equity_curve)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def run_backtest(
    config_path: str | Path = "config.live.yaml",
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 1000,
) -> tuple[BacktestSummary, list[BacktestTrade]]:
    cfg = load_config(config_path)
    engine = BacktestEngine(cfg, symbol=symbol, timeframe=timeframe, limit=limit)
    engine.load_data()
    summary = engine.run()
    return summary, engine.trades


def _print_summary(summary: BacktestSummary) -> None:
    print(json.dumps(asdict(summary), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a conservative candle-by-candle backtest using live signal logic."
    )
    parser.add_argument(
        "--config", default="config.live.yaml", help="Config file to load"
    )
    parser.add_argument(
        "--symbol", default=None, help="Trading symbol override"
    )
    parser.add_argument(
        "--timeframe", default=None, help="Base timeframe override"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Historical candles per timeframe",
    )
    parser.add_argument(
        "--trades-out",
        default=None,
        help="Optional CSV path for trade log export",
    )
    parser.add_argument(
        "--equity-out",
        default=None,
        help="Optional CSV path for equity curve export",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = BacktestEngine(
        cfg,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )
    engine.load_data()
    summary = engine.run()
    trades = engine.trades
    _print_summary(summary)

    if args.trades_out:
        trade_data = [asdict(t) for t in trades]
        pd.DataFrame(trade_data).to_csv(args.trades_out, index=False)
        print(f"Trade log written to {args.trades_out}")

    if args.equity_out:
        engine.export_equity(args.equity_out)
        print(f"Equity curve written to {args.equity_out}")


if __name__ == "__main__":
    main()
