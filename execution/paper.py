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
        self.symbol = "BTC/USDT:USDT"
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
        self.min_profit_after_fees = 0.0002
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

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    # PRIMARY EXIT: Parabolic SAR flip (SAR moves ABOVE price = trend reversal)
                    if not closed and psar is not None and hold_time > 10:
                        if psar > current_price and profit_pct >= min_net_profit:
                            # SAR flipped bearish while we are in profit — clean exit
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

                    # TTL EXIT: cut stuck positions after timeout if barely profitable
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct >= min_net_profit and profit_pct < break_even_trigger:
                        closed = True
                        exit_type = "TTL_EXIT"

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

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    # PRIMARY EXIT: Parabolic SAR flip (SAR moves BELOW price = trend reversal for short)
                    if not closed and psar is not None and hold_time > 10:
                        if psar < current_price and profit_pct >= min_net_profit:
                            # SAR flipped bullish while we are in profit — clean exit
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

                    # TTL EXIT: cut stuck positions after timeout if barely profitable
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct >= min_net_profit and profit_pct < break_even_trigger:
                        closed = True
                        exit_type = "TTL_EXIT"

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
                    exit_type = _realized_exit_type(exit_type or "TRAIL_WIN", net_pnl)
                    self.cash_usdt += (pnl - fees)

                    self._record_closed_trade(exit_type, entry, current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(trade_id, "EXIT", order_side, current_price, amount, pnl, exit_fee, t_type=exit_type)
                    self.trade_count += 1
                    self.last_status = f"PAPER {exit_type}: {side} @ ${current_price:.2f} P&L ${pnl:+,.2f}"
                    self._last_trade_ts = time.time()
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
                if (action == "SELL" and current_pos['side'] == "LONG") or (action == "BUY" and current_pos['side'] == "SHORT"):
                    hold_until = float(current_pos.get("hold_until_ts", 0.0) or 0.0)
                    if hold_until and now < hold_until:
                        self.last_status = "Veto: hold period"
                        return
                    entry_ts = float(current_pos.get("entry_ts", now))
                    age = now - entry_ts
                    profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']

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

                    logger.info(f"[REVERSAL] Banking {current_pos['side']} before {action} (P&L: {profit_pct:+.2%})")
                    order_side = 'SELL' if current_pos['side'] == 'LONG' else 'BUY'
                    pnl = (current_price - current_pos['entry']) * current_pos['amount'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) * current_pos['amount']
                    fees = (current_pos['amount'] * current_pos['entry'] * self.fee_rate) + (current_pos['amount'] * current_price * self.fee_rate)
                    exit_fee = current_pos['amount'] * current_price * self.fee_rate
                    net_pnl = pnl - fees
                    self.cash_usdt += (pnl - fees)
                    exit_type = "REVERSAL_BANK" if getattr(self, 'exit_on_reversal_only_in_profit', True) else "REVERSAL"
                    self._record_closed_trade(exit_type, current_pos['entry'], current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, exit_fee, t_type=exit_type)
                    self.active_positions = []
                    self._last_trade_ts = now
                    self.last_status = f"{exit_type}: {current_pos['side']} @ ${current_price:.5f} Net ${net_pnl:+,.2f}"
                    # Keep going so the opposite-side reversal can enter immediately.

            if len(self.active_positions) >= self.max_open_positions:
                pos_info = f"{len(self.active_positions)} pos "
                if self.active_positions:
                    p = self.active_positions[0]
                    pos_info += f"({p['side']} @ ${p['entry']:.2f}, P&L: {((current_price - p['entry']) / p['entry'] * 100 if p['side'] == 'LONG' else (p['entry'] - current_price) / p['entry'] * 100):.2f}%)"
                self.last_status = f"Max positions: {pos_info}"
                return

            balance = self.get_portfolio_value(current_price)
            if balance <= 0:
                self.last_status = "No equity available"
                return

            # Determine trade size conservatively so paper matches live risk more closely.
            configured_trade_usdt = float(getattr(self, 'fixed_trade_usdt', 0.0) or 0.0)
            if configured_trade_usdt > 0:
                trade_usdt = min(configured_trade_usdt, balance * 0.90)
            else:
                trade_usdt = balance * 0.25

            if trade_usdt < 10.0:
                self.last_status = f"Equity too low: ${balance:,.2f}"
                return

            # Use dynamic leverage based on signal confidence (PAPER VERSION)
            confidence = float(signal.get('confidence', 0.5) or 0.5)
            score = float(signal.get('score', 0.5) or 0.5)
            atr_pct = signal.get('atr_pct')  # Get ATR from signal if available
            current_leverage = self.calculate_dynamic_leverage(confidence, score, atr_pct=atr_pct)
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
            sl_price = _safe_initial_sl_price(
                pos_side,
                simulated_entry_price,
                signal.get('sl'),
                getattr(self, 'default_sl_pct', 0.0030),
                signal.get('pivot_classic'),
            )
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
