import csv
import logging
import os
import time
from collections import deque
from datetime import datetime

from .base import (
    TRADE_LOG_FILE,
    TRADE_LOG_HEADER,
    _compute_trailing_stop,
    _default_sl_price,
    _market_id_from_symbol,
    _next_trade_id_from_log,
    _normalize_futures_symbol,
    _realized_exit_type,
    _runner_emergency_tp_price,
    _safe_initial_sl_price,
    _safe_tp_price,
    _trailing_tp_hit,
)

logger = logging.getLogger(__name__)


class PaperFuturesExecution:
    def __init__(self, starting_balance_usdt: float = 1000.0, leverage: int = 5, fee_rate: float = 0.0004, max_closed_trades: int = 5000):
        self.label = "PAPER"
        self.cash_usdt = float(starting_balance_usdt)
        self.initial_balance = float(starting_balance_usdt)
        self.symbol = "AVAX/USDC:USDC"
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self.leverage = leverage
        self.fee_rate = float(fee_rate)
        self.fee_slippage_buffer_pct = 0.0
        self.fee_edge_multiplier = 1.0
        self.fixed_trade_usdt = 0.0
        self.learning_risk_multiplier = 1.0
        self.min_seconds_between_trades = 15
        self.min_seconds_before_reversal = 0
        self.reversal_min_confidence = 0.0
        self.reversal_min_score = 0.0
        self.reversal_min_net_edge_pct = 0.0
        self.same_side_reentry_cooldown_seconds = 180
        self.scale_in_enabled = False
        self.scale_in_max_steps = 2
        self.scale_in_min_pnl_pct = 0.0020
        self.scale_in_cooldown_seconds = 180
        self.scale_in_position_pct = 0.5
        self.scale_in_max_exposure_pct = 0.50
        self.scale_in_wall_buffer_pct = 0.002
        self.break_even_trigger_pct = 0.0010
        self.break_even_buffer_pct = 0.0003
        self.profit_trailing_enabled = True
        self.profit_trailing_activation_pct = self.break_even_trigger_pct
        self.trailing_tp_enabled = True
        self.trailing_tp_giveback_pct = 0.12
        self.trailing_tp_min_peak_pct = 0.0020
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self.trail_t1_gap_pct = 0.0025
        self.trail_t2_gap_pct = 0.0020
        self._last_trade_ts = 0.0
        self._last_profitable_exit_side = ""
        self._last_profitable_exit_ts = 0.0
        self._last_defensive_exit_side = ""
        self._last_defensive_exit_ts = 0.0
        self._opposite_reset_seen_after_profit = False
        self.psar_exit_min_streak = 2
        self.trade_log_file = TRADE_LOG_FILE
        self.active_positions = []
        self.last_status = "PAPER READY"
        self.closed_trades = deque(maxlen=int(max_closed_trades or 5000))
        self.trade_count = 0
        self.stats_trades = 0
        self.stats_wins = 0
        self.stats_losses = 0
        self.stats_gross = 0.0
        self.stats_fees = 0.0
        self._next_trade_id = 1
        self.max_open_positions = 1
        self.min_balance_floor = 0.0
        self.dynamic_leverage_enabled = False
        self._initial_price_set = True
        self._last_price = None
        self._current_atr_pct = 0.02
        self._current_psar = None
        self._current_psar_streak = 0
        self.min_profit_after_fees = 0.0002
        self._current_signal_snapshot = {}
        self.daily_loss_cap_pct = None
        self.disable_loss_cap = False
        self.scalp_config = {
            'tp_pct': 0.0040,
            'min_hold_seconds': 20,
            'fade_trigger_pct': 0.0060,
            'fade_exit_pct': 0.0030
        }
        self._init_trade_log()

    def get_open_orders(self, symbol: str = None) -> list:
        return []

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

    def _fetch_free_usdt(self):
        return float(self.cash_usdt)

    def _fetch_free_btc(self):
        return 0.0

    def calculate_dynamic_leverage(self, confidence: float, score: float = 0.5, atr_pct: float = None) -> float:
        """Calculate leverage based on signal confidence, score, and volatility (ATR)."""
        if not self.dynamic_leverage_enabled or not self.leverage_confidence_levels:
            return float(self.leverage)

        confidence = float(confidence or 0.0)
        score = float(score or 0.5)

        # Find appropriate leverage level based on confidence thresholds
        leverage = self.leverage_min
        for threshold in sorted(self.leverage_confidence_levels.keys()):
            if confidence >= threshold:
                leverage = self.leverage_confidence_levels[threshold]

        # Optionally adjust by score
        if self.leverage_use_score:
            score_factor = 1.0 + (score - 0.5) * self.leverage_score_weight
            leverage = leverage * score_factor

        # NEW: Volatility-based scaling using ATR
        atr_volatility_scaling = getattr(self, 'atr_volatility_scaling', False)
        if atr_volatility_scaling:
            current_atr = atr_pct if atr_pct is not None else getattr(self, '_current_atr_pct', 0.5)
            atr_reference = getattr(self, 'atr_reference_pct', 0.5)
            atr_min_multiplier = getattr(self, 'atr_min_multiplier', 0.3)

            if current_atr > 0:
                # Inverse relationship: higher volatility = lower leverage
                vol_multiplier = atr_reference / current_atr
                vol_multiplier = max(atr_min_multiplier, min(1.5, vol_multiplier))  # Cap at 1.5x
                leverage = leverage * vol_multiplier

        # Clamp to min/max
        leverage = max(self.leverage_min, min(self.leverage_max, leverage))
        return leverage

    def _same_side_reentry_veto(self, signal: dict, action: str, now: float) -> str:
        last_prof_side = str(getattr(self, "_last_profitable_exit_side", "") or "").upper()
        last_prof_ts = float(getattr(self, "_last_profitable_exit_ts", 0.0) or 0.0)
        action = str(action or "").upper()
        cooldown = float(getattr(self, "same_side_reentry_cooldown_seconds", 0) or 0)

        def _same_side(side: str) -> bool:
            return (side == "LONG" and action == "BUY") or (side == "SHORT" and action == "SELL")

        if last_prof_side in {"LONG", "SHORT"} and last_prof_ts > 0 and _same_side(last_prof_side):
            elapsed = now - last_prof_ts
            if cooldown > 0 and elapsed < cooldown:
                wait_s = int(max(1.0, cooldown - elapsed))
                return f"Veto: post-profit same-side cooldown ({wait_s}s)"

        last_def_side = str(getattr(self, "_last_defensive_exit_side", "") or "").upper()
        last_def_ts = float(getattr(self, "_last_defensive_exit_ts", 0.0) or 0.0)
        if last_def_side in {"LONG", "SHORT"} and last_def_ts > 0 and _same_side(last_def_side):
            elapsed = now - last_def_ts
            if cooldown > 0 and elapsed < cooldown:
                wait_s = int(max(1.0, cooldown - elapsed))
                return f"Veto: post-defensive same-side cooldown ({wait_s}s)"
        return ""

    def _paper_scale_in_gate(
        self,
        signal: dict,
        action: str,
        current_price: float,
        now: float,
        balance: float,
        base_trade_usdt: float,
        current_leverage: float,
    ) -> tuple[bool, str, float]:
        if not bool(getattr(self, "scale_in_enabled", False)):
            return False, "", 0.0

        action = str(action or "").upper()
        if action not in {"BUY", "SELL"}:
            return False, "", 0.0

        pos_side = "LONG" if action == "BUY" else "SHORT"
        same_side_positions = [
            p for p in self.active_positions
            if str(p.get("side", "")).upper() == pos_side
        ]
        if not same_side_positions:
            return False, "", 0.0

        if len(same_side_positions) != len(self.active_positions):
            return False, "Veto: scale-in requires single-side exposure", 0.0

        max_steps = max(1, int(getattr(self, "scale_in_max_steps", 2) or 2))
        if len(same_side_positions) >= max_steps:
            return False, f"Veto: scale-in max steps ({max_steps})", 0.0

        cooldown = float(getattr(self, "scale_in_cooldown_seconds", 0) or 0.0)
        last_entry_ts = max(float(p.get("entry_ts", now) or now) for p in same_side_positions)
        elapsed = now - last_entry_ts
        if cooldown > 0 and elapsed < cooldown:
            wait_s = int(max(1.0, cooldown - elapsed))
            return False, f"Veto: scale-in cooldown ({wait_s}s)", 0.0

        total_amount = sum(float(p.get("amount", 0.0) or 0.0) for p in same_side_positions)
        if total_amount <= 0:
            return False, "Veto: scale-in invalid position size", 0.0

        weighted_entry = sum(
            float(p.get("entry", current_price) or current_price) * float(p.get("amount", 0.0) or 0.0)
            for p in same_side_positions
        ) / total_amount
        if weighted_entry <= 0:
            return False, "Veto: scale-in invalid entry", 0.0

        pnl_pct = (
            (current_price - weighted_entry) / weighted_entry
            if pos_side == "LONG"
            else (weighted_entry - current_price) / weighted_entry
        )
        min_pnl_pct = float(getattr(self, "scale_in_min_pnl_pct", 0.0) or 0.0)
        if pnl_pct < min_pnl_pct:
            return False, f"Veto: scale-in needs profit ({pnl_pct:+.2%} < {min_pnl_pct:.2%})", 0.0

        wall_buffer_pct = float(getattr(self, "scale_in_wall_buffer_pct", 0.0) or 0.0)
        resistance = signal.get("structure_resistance")
        support = signal.get("structure_support")
        resistance_broken = bool(signal.get("resistance_broken"))
        support_broken = bool(signal.get("support_broken"))

        try:
            resistance = float(resistance) if resistance is not None else 0.0
        except (TypeError, ValueError):
            resistance = 0.0
        try:
            support = float(support) if support is not None else 0.0
        except (TypeError, ValueError):
            support = 0.0

        if pos_side == "LONG":
            if resistance > 0 and not resistance_broken:
                resistance_broken = current_price >= (resistance * (1.0 + wall_buffer_pct))
            if resistance > 0 and not resistance_broken and current_price >= (resistance * (1.0 - wall_buffer_pct)):
                return False, "Veto: scale-in near resistance", 0.0
        else:
            if support > 0 and not support_broken:
                support_broken = current_price <= (support * (1.0 - wall_buffer_pct))
            if support > 0 and not support_broken and current_price <= (support * (1.0 + wall_buffer_pct)):
                return False, "Veto: scale-in near support", 0.0

        scale_in_position_pct = float(getattr(self, "scale_in_position_pct", 0.0) or 0.0)
        if scale_in_position_pct <= 0:
            return False, "Veto: scale-in disabled", 0.0

        scale_trade_usdt = float(base_trade_usdt) * scale_in_position_pct
        if scale_trade_usdt <= 0:
            return False, "Veto: scale-in size too small", 0.0

        max_exposure_pct = float(getattr(self, "scale_in_max_exposure_pct", 0.0) or 0.0)
        if max_exposure_pct > 0 and balance > 0:
            existing_notional = sum(
                float(p.get("amount", 0.0) or 0.0) * float(current_price)
                for p in same_side_positions
            )
            new_notional = scale_trade_usdt * float(current_leverage)
            if (existing_notional + new_notional) / balance > max_exposure_pct:
                return False, f"Veto: scale-in exposure > {max_exposure_pct:.0%}", 0.0

        return True, "", scale_trade_usdt

    def _paper_scale_in_combined_sl(
        self,
        signal: dict,
        pos_side: str,
        add_entry_price: float,
        add_amount: float,
    ) -> tuple[bool, float, str]:
        same_side_positions = [
            p for p in self.active_positions
            if str(p.get("side", "")).upper() == pos_side
        ]
        if not same_side_positions:
            return False, 0.0, "Veto: scale-in missing base position"

        total_amount = float(add_amount or 0.0) + sum(float(p.get("amount", 0.0) or 0.0) for p in same_side_positions)
        if total_amount <= 0:
            return False, 0.0, "Veto: scale-in invalid combined size"

        weighted_entry = (
            float(add_entry_price or 0.0) * float(add_amount or 0.0)
            + sum(float(p.get("entry", add_entry_price) or add_entry_price) * float(p.get("amount", 0.0) or 0.0) for p in same_side_positions)
        ) / total_amount
        if weighted_entry <= 0:
            return False, 0.0, "Veto: scale-in invalid combined entry"

        max_sl_pct = abs(float(getattr(self, "max_structural_sl_pct", 0.0120) or 0.0120))
        default_sl_pct = abs(float(getattr(self, "default_sl_pct", 0.0030) or 0.0030))
        combined_sl = _safe_initial_sl_price(
            pos_side,
            weighted_entry,
            signal.get("sl"),
            default_sl_pct,
            signal.get("pivot_classic"),
            max_sl_pct=max_sl_pct,
        )
        if combined_sl <= 0:
            return False, 0.0, "Veto: scale-in invalid combined SL"

        risk_pct = abs(weighted_entry - combined_sl) / weighted_entry if weighted_entry > 0 else 0.0
        if max_sl_pct > 0 and risk_pct > (max_sl_pct + 1e-12):
            return False, 0.0, f"Veto: scale-in combined risk {risk_pct:.2%} > {max_sl_pct:.2%}"

        if pos_side == "LONG" and combined_sl >= weighted_entry:
            return False, 0.0, "Veto: scale-in combined SL above entry"
        if pos_side == "SHORT" and combined_sl <= weighted_entry:
            return False, 0.0, "Veto: scale-in combined SL below entry"

        return True, combined_sl, ""

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
        t_type: str = "",
    ):
        if event.upper() == "ENTRY":
            if not hasattr(self, "_entry_info"):
                self._entry_info = {}
            self._entry_info[int(trade_id)] = {"side": str(side).upper(), "reason": str(reason or "")}
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
                    (t_type or ""),
                ])
        except Exception as e:
            logger.error(f"Trade Log Error: {e}")

    def get_portfolio_value(self, current_price: float) -> float:
        unreal = 0.0
        for pos in self.active_positions:
            entry = float(pos['entry'])
            amount = float(pos['amount'])
            if pos['side'] == 'LONG':
                unreal += (current_price - entry) * amount
            else:
                unreal += (entry - current_price) * amount
        return float(self.cash_usdt + unreal)

    def check_risk_limits(self, current_price: float) -> bool:
        if getattr(self, "disable_loss_cap", False):
            return True
        val = self.get_portfolio_value(current_price)
        if val > 0 and val <= self.min_balance_floor:
            return False
        if self.daily_loss_cap_pct is not None and self.initial_balance > 0 and val > 0:
            drawdown = (self.initial_balance - val) / self.initial_balance
            if drawdown >= float(self.daily_loss_cap_pct):
                return False
        return True

    def _paper_runner_partial_exit(self, pos: dict, current_price: float, trade_id: int, exit_type: str = "TAKE_PROFIT") -> bool:
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        amount = float(pos.get("amount", 0.0) or 0.0)
        entry = float(pos.get("entry", current_price) or current_price)
        if side not in {"LONG", "SHORT"} or amount <= 0 or entry <= 0:
            return False

        scalp_cfg = getattr(self, "scalp_config", {}) or {}
        if not bool(scalp_cfg.get("runner_enabled", True)):
            return False
        if bool(pos.get("runner_scale_out_taken", False)):
            return False

        partial_pct = float(scalp_cfg.get("runner_partial_exit_pct", 0.0) or 0.0)
        if not (0.0 < partial_pct < 1.0):
            return False

        close_amount = amount * partial_pct
        remaining_amount = amount - close_amount
        if close_amount <= 0 or remaining_amount <= 0:
            return False

        fee_rate = float(pos.get("fee_rate", getattr(self, "fee_rate", 0.0004)) or 0.0004)
        min_profit = float(pos.get("min_profit_after_fees", getattr(self, "min_profit_after_fees", 0.0002)) or 0.0002)
        min_net_profit = (2.0 * fee_rate) + min_profit
        runner_lock = max(
            float(scalp_cfg.get("runner_min_lock_pct", 0.0) or 0.0),
            min_net_profit,
        )

        pnl = (current_price - entry) * close_amount if side == "LONG" else (entry - current_price) * close_amount
        profit_pct = (current_price - entry) / entry if side == "LONG" else (entry - current_price) / entry
        fees = (close_amount * entry * fee_rate) + (close_amount * current_price * fee_rate)
        exit_fee = close_amount * current_price * fee_rate
        net_pnl = pnl - fees
        realized_type = _realized_exit_type(exit_type or "TAKE_PROFIT", net_pnl)
        order_side = "SELL" if side == "LONG" else "BUY"

        self.cash_usdt += net_pnl
        self.stats_gross += pnl
        self.stats_fees += fees

        pos["amount"] = remaining_amount
        pos["runner_scale_out_taken"] = True
        pos["profit_runner_armed"] = True
        pos["trail_armed"] = True
        pos["fixed_take_profit_enabled"] = False
        pos["tp_price"] = 0.0
        pos["runner_partial_exit_pct"] = partial_pct
        pos["runner_remaining_amount"] = remaining_amount
        pos["highest_profit_pct"] = max(float(pos.get("highest_profit_pct", 0.0) or 0.0), profit_pct)

        if side == "LONG":
            protected_sl = entry * (1.0 + runner_lock)
            pos["sl"] = max(float(pos.get("sl", 0.0) or 0.0), protected_sl)
            if current_price > float(pos.get("highest_price", entry) or entry):
                pos["highest_price"] = current_price
        else:
            protected_sl = entry * (1.0 - runner_lock)
            pos["sl"] = min(float(pos.get("sl", 0.0) or 0.0), protected_sl)
            if current_price < float(pos.get("lowest_price", entry) or entry):
                pos["lowest_price"] = current_price

        pos["sl_pct_dist"] = abs(entry - float(pos.get("sl", entry) or entry)) / entry if entry else float(pos.get("sl_pct_dist", 0.0) or 0.0)

        self._log_trade(
            trade_id,
            "PARTIAL_EXIT",
            order_side,
            current_price,
            close_amount,
            pnl,
            exit_fee,
            t_type=realized_type,
            reason="paper runner scale-out",
        )
        self._last_trade_ts = time.time()
        self.last_status = (
            f"{realized_type}: {side} scale-out {close_amount:.8f} @ ${current_price:.5f} "
            f"Net ${net_pnl:+,.2f} | Rem {remaining_amount:.8f}"
        )
        logger.info(
            f"[PAPER_PARTIAL_EXIT] {side} scale-out {close_amount:.8f}/{amount:.8f} @ {current_price:.5f} "
            f"| Net: {net_pnl:+.4f} ({profit_pct:+.2%}) | Remaining: {remaining_amount:.8f}"
        )
        return True

    def _should_defensive_exit(self, pos: dict, current_price: float) -> bool:
        signal = getattr(self, "_current_signal_snapshot", {}) or {}
        side = str(pos.get("side", "")).upper()
        if side not in {"LONG", "SHORT"}:
            return False

        entry = float(pos.get("entry", current_price) or current_price)
        if entry <= 0:
            return False

        entry_ts = float(pos.get("entry_ts", time.time()) or time.time())
        hold_time = time.time() - entry_ts
        fee_rate = float(pos.get("fee_rate", getattr(self, "fee_rate", 0.0004)) or 0.0004)
        slippage_buffer_pct = float(getattr(self, "fee_slippage_buffer_pct", 0.0) or 0.0)
        fee_slippage_buffer_pct = max(0.0, (2.0 * fee_rate) + slippage_buffer_pct)

        def _normalized_mode() -> str:
            if signal.get("aggressive_scalp"):
                return "AGGRESSIVE_SCALP"

            mode_text = " ".join(
                str(value)
                for value in (
                    pos.get("entry_mode"),
                    signal.get("entry_mode"),
                    signal.get("mode"),
                    signal.get("intent"),
                    signal.get("reason"),
                )
                if value
            ).upper()
            if "AGGRESSIVE_SCALP" in mode_text:
                return "AGGRESSIVE_SCALP"
            if "RANGE" in mode_text:
                return "RANGE"
            if "BREAKOUT" in mode_text or "TREND" in mode_text:
                return "TREND"
            return "TREND"

        def _min_hold_seconds(mode: str) -> float:
            if mode == "AGGRESSIVE_SCALP":
                return 45.0
            if mode == "RANGE":
                return 60.0
            return 120.0

        sl_price = pos.get("sl")
        try:
            sl_price = float(sl_price) if sl_price is not None else 0.0
        except (TypeError, ValueError):
            sl_price = 0.0
        sl_distance_pct = float(pos.get("sl_pct_dist", 0.0) or 0.0)
        if sl_price > 0 and entry > 0:
            sl_distance_pct = abs(entry - sl_price) / entry

        if hold_time < _min_hold_seconds(_normalized_mode()):
            return False

        adverse_move_buffer_pct = max(fee_slippage_buffer_pct, 0.40 * sl_distance_pct)

        if side == "LONG":
            adverse_move_pct = max(0.0, (entry - current_price) / entry)
        else:
            adverse_move_pct = max(0.0, (current_price - entry) / entry)

        if adverse_move_pct < adverse_move_buffer_pct:
            return False

        bias = str(signal.get("mtf_fast_bias") or signal.get("market_bias") or "").upper()
        amount = float(pos.get("amount", 0.0) or 0.0)
        fees = (amount * entry * fee_rate) + (amount * current_price * fee_rate)
        net_pnl = ((current_price - entry) * amount - fees) if side == "LONG" else ((entry - current_price) * amount - fees)
        if net_pnl > 0:
            return False

        if side == "SHORT":
            if bias not in {"LONG_ONLY", "BULLISH"}:
                return False
            resistance = signal.get("structure_resistance")
            try:
                resistance = float(resistance) if resistance is not None else 0.0
            except (TypeError, ValueError):
                resistance = 0.0
            reclaim_level = entry if resistance <= 0 else min(entry, resistance)
            return current_price >= reclaim_level

        if bias not in {"SHORT_ONLY", "BEARISH"}:
            return False
        support = signal.get("structure_support")
        try:
            support = float(support) if support is not None else 0.0
        except (TypeError, ValueError):
            support = 0.0
        loss_level = entry if support <= 0 else max(entry, support)
        return current_price <= loss_level

    def process_orders_and_positions(self, symbol: str, current_price: float):
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self._last_price = current_price
        remaining = []
        try:
            for pos in self.active_positions:
                closed = False
                entry = pos['entry']
                side = pos['side']
                amount = pos['amount']
                trade_id = pos.get("trade_id", 0)

                exit_type = ""
                psar = getattr(self, "_current_psar", None)
                scalp_cfg = getattr(self, "scalp_config", {}) or {}
                runner_enabled = bool(scalp_cfg.get("runner_enabled", True))

                if side == 'LONG':
                    if current_price > float(pos.get('highest_price', entry) or entry):
                        pos['highest_price'] = current_price

                    profit_pct = (current_price - entry) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', getattr(self, 'min_profit_after_fees', 0.0002)))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit
                    trail_activation = float(pos.get('profit_trailing_activation_pct', pos.get('break_even_trigger_pct', 0.0030)) or 0.0030)
                    if bool(pos.get("profit_trailing_enabled", True)):
                        trail_activation = max(trail_activation, float(pos.get('break_even_trigger_pct', 0.0030) or 0.0030))

                    if not closed and self._should_defensive_exit(pos, current_price):
                        closed = True
                        exit_type = "DEFENSIVE_EXIT"

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    # TAKE PROFIT: explicit TP price hit
                    _tp_price = float(pos.get('tp_price') or 0.0)
                    if not closed and _tp_price > 0 and current_price >= _tp_price:
                        if runner_enabled and not bool(pos.get("runner_scale_out_taken", False)):
                            if self._paper_runner_partial_exit(pos, current_price, trade_id, exit_type="TAKE_PROFIT"):
                                remaining.append(pos)
                                continue
                            closed = True
                            exit_type = "TAKE_PROFIT"
                        else:
                            closed = True
                            exit_type = "TAKE_PROFIT"

                    # PRIMARY EXIT: Parabolic SAR flip (streak-confirmed reversal)
                    if not closed and psar is not None and hold_time > 60:
                        psar_streak = int(getattr(self, "_current_psar_streak", 0) or 0)
                        min_psar_streak = max(1, int(getattr(self, "psar_exit_min_streak", 2) or 2))
                        if (
                            psar > current_price
                            and profit_pct >= min_net_profit
                            and psar_streak <= -min_psar_streak
                        ):
                            # SAR flipped bearish with enough streak confirmation while profitable.
                            closed = True
                            exit_type = "PSAR_EXIT"

                    # TRAILING STOP via PSAR: when in profit, trail SL to PSAR
                    if not closed and psar is not None and psar < current_price and profit_pct >= trail_activation:
                        # PSAR is below price (bullish) — use it as dynamic trailing stop
                        if psar > float(pos['sl']):
                            pos['sl'] = psar
                            logger.info(f"[PSAR_TRAIL] LONG SL moved to PSAR {psar:.5f}")

                    # BREAK-EVEN LOCK: once trade reaches break-even trigger, lock SL above entry
                    break_even_trigger = float(pos.get('break_even_trigger_pct', 0.0030))
                    if not closed and profit_pct >= break_even_trigger:
                        min_sl_price = entry * (1 + (2.0 * fee_rate) + min_profit)
                        if min_sl_price > float(pos['sl']):
                            pos['sl'] = min_sl_price

                    # TTL EXIT: cut stuck positions after timeout (losers AND barely-profitable)
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds:
                        if profit_pct >= 0:
                            min_sl_price = entry * (1 + (2.0 * fee_rate) + min_profit)
                            if min_sl_price > float(pos['sl']):
                                pos['sl'] = min_sl_price
                        elif profit_pct < break_even_trigger:  # not yet locked in profit — cut it
                            closed = True
                            exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        if new_sl > float(pos['sl']):
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Support Lock)")

                    # STRUCTURAL SL CHECK: price hit stop loss
                    if not closed and current_price <= pos['sl']:
                        closed = True
                        profit_pct = (current_price - entry) / entry

                else: # SHORT
                    if current_price < float(pos.get('lowest_price', entry) or entry):
                        pos['lowest_price'] = current_price

                    profit_pct = (entry - current_price) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', getattr(self, 'min_profit_after_fees', 0.0002)))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit
                    trail_activation = float(pos.get('profit_trailing_activation_pct', pos.get('break_even_trigger_pct', 0.0030)) or 0.0030)
                    if bool(pos.get("profit_trailing_enabled", True)):
                        trail_activation = max(trail_activation, float(pos.get('break_even_trigger_pct', 0.0030) or 0.0030))

                    if not closed and self._should_defensive_exit(pos, current_price):
                        closed = True
                        exit_type = "DEFENSIVE_EXIT"

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    # TAKE PROFIT: explicit TP price hit
                    _tp_price = float(pos.get('tp_price') or 0.0)
                    if not closed and _tp_price > 0 and current_price <= _tp_price:
                        if runner_enabled and not bool(pos.get("runner_scale_out_taken", False)):
                            if self._paper_runner_partial_exit(pos, current_price, trade_id, exit_type="TAKE_PROFIT"):
                                remaining.append(pos)
                                continue
                            closed = True
                            exit_type = "TAKE_PROFIT"
                        else:
                            closed = True
                            exit_type = "TAKE_PROFIT"

                    # PRIMARY EXIT: Parabolic SAR flip (streak-confirmed reversal)
                    if not closed and psar is not None and hold_time > 60:
                        psar_streak = int(getattr(self, "_current_psar_streak", 0) or 0)
                        min_psar_streak = max(1, int(getattr(self, "psar_exit_min_streak", 2) or 2))
                        if (
                            psar < current_price
                            and profit_pct >= min_net_profit
                            and psar_streak >= min_psar_streak
                        ):
                            # SAR flipped bullish with enough streak confirmation while profitable.
                            closed = True
                            exit_type = "PSAR_EXIT"

                    # TRAILING STOP via PSAR: when in profit, trail SL to PSAR
                    if not closed and psar is not None and psar > current_price and profit_pct >= trail_activation:
                        # PSAR is above price (bearish) — use it as dynamic trailing stop
                        if psar < float(pos['sl']):
                            pos['sl'] = psar
                            logger.info(f"[PSAR_TRAIL] SHORT SL moved to PSAR {psar:.5f}")

                    # BREAK-EVEN LOCK: once trade reaches break-even trigger, lock SL below entry
                    break_even_trigger = float(pos.get('break_even_trigger_pct', 0.0030))
                    if not closed and profit_pct >= break_even_trigger:
                        max_sl_price = entry * (1 - (2.0 * fee_rate) - min_profit)
                        if max_sl_price < float(pos['sl']):
                            pos['sl'] = max_sl_price

                    # TTL EXIT: cut stuck positions after timeout (losers AND barely-profitable)
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds:
                        if profit_pct >= 0:
                            max_sl_price = entry * (1 - (2.0 * fee_rate) - min_profit)
                            if max_sl_price < float(pos['sl']):
                                pos['sl'] = max_sl_price
                        elif profit_pct < break_even_trigger:  # not yet locked in profit — cut it
                            closed = True
                            exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        if new_sl < float(pos['sl']):
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Resistance Lock)")

                    # STRUCTURAL SL CHECK: price hit stop loss
                    if not closed and current_price >= pos['sl']:
                        closed = True
                        profit_pct = (entry - current_price) / entry

                if closed:
                    profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
                    order_side = 'SELL' if side == 'LONG' else 'BUY'

                    pnl = (current_price - entry) * amount if side == 'LONG' else (entry - current_price) * amount
                    fees = (amount * entry * self.fee_rate) + (amount * current_price * self.fee_rate)
                    exit_fee = amount * current_price * self.fee_rate
                    net_pnl = pnl - fees
                    exit_type = exit_type if exit_type == "DEFENSIVE_EXIT" else _realized_exit_type(exit_type or "TRAIL_WIN", net_pnl)
                    self.cash_usdt += (pnl - fees)

                    self._record_closed_trade(exit_type, entry, current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(trade_id, "EXIT", order_side, current_price, amount, pnl, exit_fee, t_type=exit_type)
                    self.trade_count += 1
                    self.last_status = f"PAPER {exit_type}: {side} @ ${current_price:.2f} P&L ${pnl:+,.2f}"
                    self._last_trade_ts = time.time()
                    if exit_type == "DEFENSIVE_EXIT":
                        self._last_defensive_exit_side = side
                        self._last_defensive_exit_ts = self._last_trade_ts
                else:
                    remaining.append(pos)
            self.active_positions = remaining
        except Exception:
            logger.exception("Paper Process Error")
            self.last_status = "Paper process error (check logs)"

    def place_limit_order(self, signal: dict, symbol: str, current_price: float):
        if getattr(self, "paused", False):
            self.last_status = "Trading PAUSED"
            return
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        action = signal['action']
        if action == "HOLD":
            self.last_status = "Signal HOLD"
            return

        try:
            now = time.time()
            if self.min_seconds_between_trades and (now - float(self._last_trade_ts)) < float(self.min_seconds_between_trades):
                self.last_status = f"Cooldown ({int(self.min_seconds_between_trades)}s)"
                return

            # Fee-aware minimum edge filter (TP distance must clear estimated costs)
            try:
                tp = float(signal.get("tp", 0.0) or 0.0)
                expected_tp_pct = abs(tp - float(current_price)) / float(current_price) if current_price else 0.0
                roundtrip_cost_pct = (2.0 * float(self.fee_rate)) + float(self.fee_slippage_buffer_pct)
                if roundtrip_cost_pct > 0 and expected_tp_pct < (float(self.fee_edge_multiplier) * roundtrip_cost_pct):
                    self.last_status = "Veto: edge < fees"
                    return
            except Exception:
                pass

            if self.active_positions:
                current_pos = self.active_positions[0]
                current_side = str(current_pos.get("side", "")).upper()
                if (action == "SELL" and current_side == "LONG") or (action == "BUY" and current_side == "SHORT"):
                    reversal_positions = [
                        p for p in self.active_positions
                        if str(p.get("side", "")).upper() == current_side
                    ]
                    if not reversal_positions:
                        return

                    hold_until = max(float(p.get("hold_until_ts", 0.0) or 0.0) for p in reversal_positions)
                    if hold_until and now < hold_until:
                        self.last_status = "Veto: hold period"
                        return

                    entry_ts = float(current_pos.get("entry_ts", now))
                    age = now - entry_ts
                    total_amount = sum(float(p.get("amount", 0.0) or 0.0) for p in reversal_positions)
                    if total_amount <= 0:
                        self.last_status = "Veto: reversal invalid size"
                        return

                    weighted_entry = sum(
                        float(p.get("entry", current_price) or current_price) * float(p.get("amount", 0.0) or 0.0)
                        for p in reversal_positions
                    ) / total_amount
                    profit_pct = (
                        (current_price - weighted_entry) / weighted_entry
                        if current_side == "LONG"
                        else (weighted_entry - current_price) / weighted_entry
                    )

                    round_trip_fee_pct = 2.0 * float(self.fee_rate)
                    reentry_fee_pct = float(self.fee_rate)
                    slippage_pct = float(getattr(self, "fee_slippage_buffer_pct", 0.0) or 0.0)
                    min_profit_after_fees = float(getattr(self, 'min_profit_after_fees', 0.0002))
                    min_net_profit = max(
                        float(getattr(self, "reversal_min_net_edge_pct", 0.0002) or 0.0002),
                        round_trip_fee_pct + reentry_fee_pct + (2.0 * slippage_pct) + min_profit_after_fees,
                    )
                    if profit_pct < min_net_profit:
                        self.last_status = f"Veto: reversal < net edge ({profit_pct:+.2%} < {min_net_profit:.2%})"
                        return

                    if self.min_seconds_before_reversal and age < float(self.min_seconds_before_reversal):
                        self.last_status = f"Veto: reversal cooldown ({int(self.min_seconds_before_reversal)}s)"
                        return

                    conf = float(signal.get("confidence", 0.0) or 0.0)
                    score = float(signal.get("score", 0.0) or 0.0)
                    if conf < float(self.reversal_min_confidence) or abs(score) < float(self.reversal_min_score):
                        self.last_status = "Veto: reversal weak"
                        return

                    logger.info(
                        f"[REVERSAL] Banking {len(reversal_positions)} {current_side} legs before {action} "
                        f"(P&L: {profit_pct:+.2%})"
                    )
                    order_side = 'SELL' if current_side == 'LONG' else 'BUY'
                    exit_type = "REVERSAL_BANK" if getattr(self, 'exit_on_reversal_only_in_profit', True) else "REVERSAL"
                    total_net_pnl = 0.0
                    total_pnl = 0.0
                    total_fees = 0.0
                    for pos in reversal_positions:
                        pos_entry = float(pos.get("entry", current_price) or current_price)
                        pos_amount = float(pos.get("amount", 0.0) or 0.0)
                        pos_pnl = (current_price - pos_entry) * pos_amount if current_side == "LONG" else (pos_entry - current_price) * pos_amount
                        pos_fees = (pos_amount * pos_entry * self.fee_rate) + (pos_amount * current_price * self.fee_rate)
                        pos_exit_fee = pos_amount * current_price * self.fee_rate
                        pos_profit_pct = (
                            (current_price - pos_entry) / pos_entry * 100.0
                            if current_side == "LONG"
                            else (pos_entry - current_price) / pos_entry * 100.0
                        )
                        pos_net_pnl = pos_pnl - pos_fees
                        total_pnl += pos_pnl
                        total_fees += pos_fees
                        total_net_pnl += pos_net_pnl
                        self.cash_usdt += pos_net_pnl
                        self._record_closed_trade(exit_type, pos_entry, current_price, pos_pnl, pos_profit_pct, pos_fees)
                        self._log_trade(
                            pos.get("trade_id", 0),
                            "EXIT",
                            order_side,
                            current_price,
                            pos_amount,
                            pos_pnl,
                            pos_exit_fee,
                            t_type=exit_type,
                        )
                    self.active_positions = []
                    self._last_trade_ts = now
                    self.last_status = f"{exit_type}: {current_side} x{len(reversal_positions)} @ ${current_price:.5f} Net ${total_net_pnl:+,.2f}"
                    if total_net_pnl > 0:
                        self._last_profitable_exit_side = current_side
                        self._last_profitable_exit_ts = now
                        self._opposite_reset_seen_after_profit = False
                    # Keep going so the opposite-side reversal can enter immediately.

            balance = self.get_portfolio_value(current_price)
            if balance <= 0:
                self.last_status = "No equity available"
                return

            configured_trade_usdt = float(getattr(self, 'fixed_trade_usdt', 0.0) or 0.0)
            if configured_trade_usdt > 0:
                base_trade_usdt = min(configured_trade_usdt, balance * 0.90)
            else:
                base_trade_usdt = balance * 0.25

            if base_trade_usdt < 10.0:
                self.last_status = f"Equity too low: ${balance:,.2f}"
                return

            confidence = float(signal.get('confidence', 0.5) or 0.5)
            score = float(signal.get('score', 0.5) or 0.5)
            atr_pct = signal.get('atr_pct')
            current_leverage = self.calculate_dynamic_leverage(confidence, score, atr_pct=atr_pct)

            scale_in_allowed = False
            scale_in_trade_usdt = 0.0
            if self.active_positions:
                current_side = str(self.active_positions[0].get("side", "")).upper()
                same_side_action = (action == "BUY" and current_side == "LONG") or (action == "SELL" and current_side == "SHORT")
                if same_side_action:
                    scale_in_allowed, scale_in_veto, scale_in_trade_usdt = self._paper_scale_in_gate(
                        signal,
                        action,
                        current_price,
                        now,
                        balance,
                        base_trade_usdt,
                        current_leverage,
                    )
                    if not scale_in_allowed:
                        self.last_status = scale_in_veto
                        return

            if not scale_in_allowed:
                same_side_veto = self._same_side_reentry_veto(signal, action, now)
                if same_side_veto:
                    self.last_status = same_side_veto
                    return

            if len(self.active_positions) >= self.max_open_positions and not scale_in_allowed:
                pos_info = f"{len(self.active_positions)} pos "
                if self.active_positions:
                    p = self.active_positions[0]
                    pos_info += f"({p['side']} @ ${p['entry']:.2f}, P&L: {((current_price - p['entry']) / p['entry'] * 100 if p['side'] == 'LONG' else (p['entry'] - current_price) / p['entry'] * 100):.2f}%)"
                self.last_status = f"Max positions: {pos_info}"
                return

            trade_usdt = scale_in_trade_usdt if scale_in_allowed else base_trade_usdt
            if trade_usdt < 10.0:
                self.last_status = f"Trade too small: ${trade_usdt:.2f}"
                return
            amount = (trade_usdt * current_leverage) / current_price
            trade_id = self._next_trade_id
            self._next_trade_id += 1

            leverage_info = f" (conf={confidence:.1%}→{current_leverage:.1f}x)" if self.dynamic_leverage_enabled else ""
            base_asset = str(self.symbol).split('/')[0] if '/' in str(self.symbol) else "units"
            logger.info(f"Paper {action}: {amount:.6f} {base_asset} @ {current_price:.5f}{leverage_info}")
            self.last_status = f"PAPER {action}: {amount:.6f} {leverage_info}"

            use_limit = getattr(self, 'use_limit_orders', False)
            simulated_entry_price = float(signal.get('entry', current_price)) if use_limit else current_price
            pos_side = 'LONG' if action == "BUY" else 'SHORT'
            max_structural_sl_pct = float(getattr(self, 'max_structural_sl_pct', 0.0120) or 0.0120)
            sl_price = _safe_initial_sl_price(
                pos_side,
                simulated_entry_price,
                signal.get('sl'),
                getattr(self, 'default_sl_pct', 0.0030),
                signal.get('pivot_classic'),
                max_sl_pct=max_structural_sl_pct,
            )
            if scale_in_allowed:
                sl_ok, combined_sl_price, sl_veto = self._paper_scale_in_combined_sl(
                    signal,
                    pos_side,
                    simulated_entry_price,
                    amount,
                )
                if not sl_ok:
                    self.last_status = sl_veto
                    return
                sl_price = float(combined_sl_price)

            sl_dist = abs(simulated_entry_price - sl_price) / simulated_entry_price if simulated_entry_price else float(getattr(self, 'default_sl_pct', 0.0030))
            try:
                tp_price = float(signal.get('tp')) if signal.get('tp') else None
            except (TypeError, ValueError):
                tp_price = None

            self.active_positions.append({
                'trade_id': trade_id,
                'side': pos_side,
                'entry': simulated_entry_price,
                'amount': amount,
                'trade_usdt': float(trade_usdt),
                'effective_leverage': 1.0,
                'entry_ts': now,
                'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                'highest_price': simulated_entry_price if action == "BUY" else 0,
                'lowest_price': simulated_entry_price if action == "SELL" else 0,
                'highest_profit_pct': 0.0,
                'sl_pct_dist': sl_dist,
                'fee_rate': float(self.fee_rate),
                'min_profit_after_fees': float(getattr(self, 'min_profit_after_fees', 0.0002)),
                'break_even_trigger_pct': float(self.break_even_trigger_pct),
                'break_even_buffer_pct': float(self.break_even_buffer_pct),
                'profit_trailing_enabled': bool(getattr(self, 'profit_trailing_enabled', True)),
                'profit_trailing_activation_pct': float(getattr(self, 'profit_trailing_activation_pct', self.break_even_trigger_pct)),
                'trailing_tp_enabled': bool(getattr(self, 'trailing_tp_enabled', True)),
                'trailing_tp_giveback_pct': float(getattr(self, 'trailing_tp_giveback_pct', 0.12)),
                'trailing_tp_min_peak_pct': float(getattr(self, 'trailing_tp_min_peak_pct', getattr(self, 'profit_trailing_activation_pct', 0.0020))),
                'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
                'trail_t1_gap_pct': float(getattr(self, 'trail_t1_gap_pct', 0.0025)),
                'trail_t2_gap_pct': float(getattr(self, 'trail_t2_gap_pct', 0.0020)),
                'trail_armed': False,
                'sl': sl_price,
                'tp_price': tp_price,
                'structure_support': signal.get('structure_support'),
                'structure_resistance': signal.get('structure_resistance'),
                'tp_target': float(signal.get('tp_target', 0.0) or 0.0),
            })
            if scale_in_allowed:
                for pos in self.active_positions:
                    if str(pos.get("side", "")).upper() != pos_side:
                        continue
                    entry_px = float(pos.get("entry", simulated_entry_price) or simulated_entry_price)
                    pos["sl"] = sl_price
                    pos["sl_pct_dist"] = abs(entry_px - sl_price) / entry_px if entry_px > 0 else float(getattr(self, 'default_sl_pct', 0.0030))

            self.trade_count += 1
            self.last_status = f"PAPER ENTRY {action} {amount:.6f}"
            self._last_trade_ts = now
            entry_fee = amount * simulated_entry_price * self.fee_rate
            self._log_trade(
                trade_id,
                "ENTRY",
                action,
                simulated_entry_price,
                amount,
                fees=entry_fee,
                score=float(signal.get("score", 0.0) or 0.0),
                confidence=float(signal.get("confidence", 0.0) or 0.0),
                reason=str(signal.get("reason", "") or ""),
            )
        except Exception as e:
            logger.error(f"Paper Placement Error: {e}")
            self.last_status = f"Paper placement error: {e}"

    def close_all_positions(self, symbol: str):
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        if not self.active_positions:
            return
        if self._last_price is None:
            self.active_positions = []
            self.last_status = "PAPER CLOSE (no price)"
            return

        current_price = float(self._last_price)
        for pos in list(self.active_positions):
            entry = float(pos['entry'])
            amount = float(pos['amount'])
            side = pos['side']
            trade_id = pos.get("trade_id", 0)

            order_side = 'SELL' if side == 'LONG' else 'BUY'
            pnl = (current_price - entry) * amount if side == 'LONG' else (entry - current_price) * amount
            fees = (amount * entry * self.fee_rate) + (amount * current_price * self.fee_rate)
            exit_fee = amount * current_price * self.fee_rate
            self.cash_usdt += (pnl - fees)

            profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
            self._record_closed_trade("MANUAL_CLOSE", entry, current_price, pnl, profit_pct * 100, fees)
            self._log_trade(trade_id, "EXIT", order_side, current_price, amount, pnl, exit_fee, t_type="MANUAL_CLOSE")
            self.trade_count += 1

        self.active_positions = []
        self.last_status = "PAPER CLOSE ALL"
