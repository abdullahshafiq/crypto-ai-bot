import ccxt
import csv
import logging
import os
import time
from collections import deque
from datetime import datetime

from .base import (
    TRADE_LOG_FILE,
    _compute_trailing_stop,
    _exchange_flag_true,
    _market_id_from_symbol,
    _next_trade_id_from_log,
    _order_fill_price,
    _order_id,
    _order_trigger_price,
    _order_type,
    _maker_entry_price,
    _realized_exit_type,
    _normalize_spot_symbol,
    _safe_initial_sl_price,
)

from .futures import BinanceFuturesExecution

logger = logging.getLogger(__name__)


class BinanceSpotExecution(BinanceFuturesExecution):
    def __init__(self, api_key: str, api_secret: str, max_closed_trades: int = 5000, is_demo: bool = False):
        self.is_demo = is_demo
        self.label = "BINANCE SPOT DEMO" if is_demo else "BINANCE SPOT LIVE"
        self.fee_rate = 0.0010
        self.fee_slippage_buffer_pct = 0.0
        self.fee_edge_multiplier = 1.0
        self.fixed_trade_usdt = 0.0
        self.spot_balance_pct = 0.20
        self.spot_reserve_pct = 0.30
        self.spot_max_layers = 3
        self.learning_risk_multiplier = 1.0
        self.min_seconds_between_trades = 0
        self.min_seconds_before_reversal = 0
        self.reversal_min_confidence = 0.0
        self.reversal_min_score = 0.0
        self.reversal_min_net_edge_pct = 0.0
        self.break_even_trigger_pct = 0.0010
        self.break_even_buffer_pct = 0.0002
        self.profit_trailing_enabled = True
        self.profit_trailing_activation_pct = self.break_even_trigger_pct
        self.trailing_tp_enabled = True
        self.trailing_tp_giveback_pct = 0.12
        self.trailing_tp_min_peak_pct = 0.0020
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self.trail_t1_gap_pct = 0.0025
        self.trail_t2_gap_pct = 0.0020
        self.default_sl_pct = 0.0030
        self.exit_on_reversal_only_in_profit = True
        self._last_trade_ts = 0.0
        self.trade_log_file = "trade_log_spot.csv"
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        if self.is_demo:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception as e:
                logger.warning(f"Spot sandbox setup note: {e}")

        self.symbol = "DOGE/USDC"
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self.leverage = 1
        self.active_positions = []
        if max_closed_trades < 100:
            max_closed_trades = 100
        self.closed_trades = deque(maxlen=int(max_closed_trades))
        self.stats_trades = 0
        self.stats_wins = 0
        self.stats_losses = 0
        self.stats_gross = 0.0
        self.stats_fees = 0.0
        self.trade_count = 0
        self._next_trade_id = 1
        self.pending_entry = None
        self.pending_exit = None
        self.pending_entry_ttl_seconds = 20
        self.pending_exit_ttl_seconds = 20
        self._entry_block_until_ts = 0.0
        self.max_open_positions = 1
        self.min_balance_floor = 0.0
        self.daily_loss_cap_pct = None
        self.dynamic_leverage_enabled = False
        self.leverage_min = 1.0
        self.leverage_max = 1.0
        self.leverage_confidence_levels = {}
        self.leverage_use_score = False
        self.leverage_score_weight = 0.0
        self.atr_volatility_scaling = False
        self.atr_reference_pct = 0.02
        self.atr_min_multiplier = 1.0
        self.min_profit_after_fees = 0.0002
        self.use_native_trailing_stop = False
        self.trailing_stop_callback = 0.005
        self._last_position_sync_ok = True
        self._last_flat_order_cleanup_ts = 0.0
        self.scalp_config = {
            'tp_pct': 0.0040,
            'min_hold_seconds': 20,
            'fade_trigger_pct': 0.0060,
            'fade_exit_pct': 0.0030
        }
        self.last_status = "SPOT INIT"
        self._last_price = None
        self.initial_balance = 0.0
        self._initial_price_set = False
        self.session_start = time.time()
        self._current_atr_pct = 0.02
        self.spot_mode = "grid"
        self.layer_quote_pct = 0.20
        self.reserve_quote_pct = 0.30
        self.buy_near_support_pct = 0.0020
        self.sell_near_resistance_pct = 0.0020
        self.layer_spacing_pct = 0.0030
        self.emergency_break_pct = 0.0040
        self.min_take_profit_pct = 0.0035
        self.max_spot_layers = self.spot_max_layers
        self._init_trade_log()
        logger.info("Binance Spot execution initialized.")

    def _base_quote_assets(self):
        symbol = _normalize_spot_symbol(getattr(self, "symbol", "DOGE/USDC"))
        if "/" not in symbol:
            return symbol, "USDC"
        base, quote = symbol.split("/", 1)
        return base.upper(), quote.upper()

    def _spot_layer_count(self) -> int:
        return sum(1 for pos in (self.active_positions or []) if str(pos.get("side", "")).upper() == "LONG")

    def _spot_last_entry_price(self) -> float:
        if not self.active_positions:
            return 0.0
        try:
            return max(float(pos.get("entry", 0.0) or 0.0) for pos in self.active_positions)
        except Exception:
            return 0.0

    def _spot_support_price(self, signal: dict) -> float:
        try:
            return float(signal.get("structure_support", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _spot_resistance_price(self, signal: dict) -> float:
        try:
            return float(signal.get("structure_resistance", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _spot_near_support(self, current_price: float, support: float) -> bool:
        if support <= 0 or current_price <= 0:
            return False
        return current_price <= support * (1 + float(self.buy_near_support_pct or 0.0))

    def _spot_layer_eligible(self, current_price: float, support: float) -> bool:
        if self.spot_mode != "grid":
            return True
        if not self._spot_near_support(current_price, support):
            return False
        last_entry = self._spot_last_entry_price()
        if last_entry > 0 and current_price > last_entry * (1 - float(self.layer_spacing_pct or 0.0)):
            return False
        return True

    def _spot_tp_price(self, entry: float, resistance: float) -> float:
        entry = float(entry or 0.0)
        resistance = float(resistance or 0.0)
        if entry <= 0:
            return 0.0
        floor_tp = entry * (1 + float(self.min_take_profit_pct or 0.0))
        if resistance > entry:
            tp = resistance * (1 - float(self.sell_near_resistance_pct or 0.0))
            return max(tp, floor_tp)
        return floor_tp

    def _spot_emergency_price(self, support: float) -> float:
        support = float(support or 0.0)
        if support <= 0:
            return 0.0
        return support * (1 - float(self.emergency_break_pct or 0.0))

    def _spot_place_tp_order(self, pos: dict, current_price: float) -> bool:
        if not isinstance(pos, dict):
            return False
        if pos.get("tp_order_id"):
            return True
        tp_price = float(pos.get("tp_price", 0.0) or 0.0)
        amount = float(pos.get("amount", 0.0) or 0.0)
        if tp_price <= 0 or amount <= 0:
            return False
        try:
            amount_s = self.exchange.amount_to_precision(self.symbol, amount)
            price_s = self.exchange.price_to_precision(self.symbol, tp_price)
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="LIMIT_MAKER",
                side="SELL",
                amount=float(amount_s),
                price=float(price_s),
                params={"postOnly": True},
            )
            pos["tp_order_id"] = str(order.get("id") or (order.get("info", {}) or {}).get("orderId") or "")
            pos["tp_order_price"] = float(price_s)
            logger.info(f"[SPOT] TP maker placed @ {price_s}")
            return True
        except Exception as e:
            logger.warning(f"Spot TP order failed: {e}")
            return False

    def _spot_close_position_emergency(self, pos: dict, current_price: float, reason: str):
        try:
            tp_id = str(pos.get("tp_order_id") or "")
            if tp_id:
                try:
                    self.exchange.cancel_order(tp_id, self.symbol)
                except Exception:
                    pass
            amount = float(pos.get("amount", 0.0) or 0.0)
            if amount <= 0:
                return False
            self._cleanup_trade_orders(self.symbol, pos)
            order_resp = self.exchange.create_market_order(self.symbol, "SELL", amount)
            fill_price = _order_fill_price(order_resp, current_price)
            entry = float(pos.get("entry", fill_price) or fill_price)
            pnl = (fill_price - entry) * amount
            profit_pct = (fill_price - entry) / entry if entry else 0.0
            fees = (amount * entry * self.fee_rate) + (amount * fill_price * self.fee_rate)
            exit_fee = amount * fill_price * self.fee_rate
            net_pnl = pnl - fees
            exit_type = _realized_exit_type(reason, net_pnl)
            self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
            self._log_trade(pos.get("trade_id", 0), "EXIT", "SELL", fill_price, amount, pnl, exit_fee, t_type=exit_type)
            self.trade_count += 1
            self._last_trade_ts = time.time()
            self.last_status = f"SPOT {exit_type}: LONG @ ${fill_price:.5f} Net ${net_pnl:+,.2f}"
            return True
        except Exception as e:
            logger.warning(f"Spot emergency close failed; keeping position active: {e}")
            self.last_status = "Spot emergency exit failed"
            return False

    def _spot_update_trailing_stop(self, pos: dict, current_price: float) -> bool:
        entry = float(pos.get("entry", current_price) or current_price)
        if entry <= 0 or current_price <= 0:
            return False

        if current_price > float(pos.get("highest_price", entry) or entry):
            pos["highest_price"] = float(current_price)

        best_price = float(pos.get("highest_price", entry) or entry)
        profit_pct = (current_price - entry) / entry
        best_profit_pct = (best_price - entry) / entry
        pos["highest_profit_pct"] = max(float(pos.get("highest_profit_pct", 0.0) or 0.0), best_profit_pct)
        pos["lowest_profit_pct"] = min(float(pos.get("lowest_profit_pct", 0.0) or 0.0), profit_pct)

        break_even_trigger = float(pos.get("break_even_trigger_pct", self.break_even_trigger_pct) or self.break_even_trigger_pct)
        trail_activation = float(pos.get("profit_trailing_activation_pct", break_even_trigger) or break_even_trigger)
        if bool(pos.get("profit_trailing_enabled", True)):
            trail_activation = max(trail_activation, break_even_trigger)
        break_even_buffer = float(pos.get("break_even_buffer_pct", self.break_even_buffer_pct) or self.break_even_buffer_pct)
        if not bool(pos.get("trail_armed", False)) and profit_pct >= trail_activation and profit_pct > 0:
            pos["trail_armed"] = True

        if not bool(pos.get("trail_armed", False)):
            return False

        new_sl = _compute_trailing_stop(pos, current_price)
        profit_floor = entry * (1 + break_even_buffer)
        new_sl = max(float(new_sl), profit_floor)
        existing_sl = float(pos.get("sl", 0.0) or 0.0)
        pos["sl"] = max(existing_sl, new_sl)
        pos["trail_mode"] = "LOCAL/PROFIT"
        return current_price <= float(pos["sl"])

    def _fetch_free_usdt(self):
        try:
            _, quote = self._base_quote_assets()
            balance = self.exchange.fetch_balance()
            return float(balance.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception as e:
            logger.error(f"Spot balance fetch error: {e}")
            self.last_status = f"Spot balance error: {e}"
            return 0.0

    def get_portfolio_value(self, current_price: float) -> float:
        try:
            base, quote = self._base_quote_assets()
            balance = self.exchange.fetch_balance()
            quote_total = float(balance.get('total', {}).get(quote, 0.0) or 0.0)
            base_total = float(balance.get('total', {}).get(base, 0.0) or 0.0)
            total_equity = quote_total + (base_total * float(current_price or 0.0))
            if not self._initial_price_set and total_equity > 0:
                self.initial_balance = total_equity
                self._initial_price_set = True
                logger.info(f"Initial Spot Session Equity: ${self.initial_balance:,.2f}")
            return total_equity
        except Exception as e:
            logger.error(f"Spot equity fetch error: {e}")
            self.last_status = f"Spot equity error: {e}"
            return 0.0

    def process_orders_and_positions(self, symbol: str, current_price: float):
        self.symbol = _normalize_spot_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self._last_price = current_price
        now = time.time()

        pending_entry = getattr(self, "pending_entry", None)
        if pending_entry:
            try:
                order_id = str(pending_entry.get("order_id") or "")
                age = now - float(pending_entry.get("ts", now) or now)
                order = self.exchange.fetch_order(order_id, self.symbol) if order_id else {}
                status = str(order.get("status", "") or "").lower()
                filled = float(order.get("filled", 0.0) or 0.0)
                expected_amount = float(pending_entry.get("amount", 0.0) or 0.0)
                if status == "closed" or (expected_amount > 0 and filled >= expected_amount * 0.999):
                    fill_price = _order_fill_price(order, float(pending_entry.get("price", current_price) or current_price))
                    support = float(pending_entry.get("structure_support", 0.0) or 0.0)
                    resistance = float(pending_entry.get("structure_resistance", 0.0) or 0.0)
                    sl_price = _safe_initial_sl_price("LONG", fill_price, pending_entry.get("sl"), self.default_sl_pct)
                    tp_price = self._spot_tp_price(fill_price, resistance)
                    self.active_positions.append({
                        'trade_id': int(pending_entry.get("trade_id", 0) or 0),
                        'side': "LONG",
                        'entry': fill_price,
                        'amount': float(order.get("filled", expected_amount) or expected_amount),
                        'entry_ts': now,
                        'hold_until_ts': float(pending_entry.get("hold_until_ts", 0.0) or 0.0),
                        'highest_price': fill_price,
                        'lowest_price': 0,
                        'highest_profit_pct': 0.0,
                        'lowest_profit_pct': 0.0,
                        'sl_pct_dist': abs(fill_price - sl_price) / fill_price if fill_price else self.default_sl_pct,
                        'fee_rate': float(self.fee_rate),
                        'min_profit_after_fees': float(self.min_profit_after_fees),
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
                        'tp_order_id': "",
                        'structure_support': support,
                        'structure_resistance': resistance,
                    })
                    self._log_trade(
                        int(pending_entry.get("trade_id", 0) or 0),
                        "ENTRY",
                        "BUY",
                        fill_price,
                        float(order.get("filled", expected_amount) or expected_amount),
                        fees=float(order.get("filled", expected_amount) or expected_amount) * fill_price * self.fee_rate,
                        score=float(pending_entry.get("score", 0.0) or 0.0),
                        confidence=float(pending_entry.get("confidence", 0.0) or 0.0),
                        reason=str(pending_entry.get("reason", "") or ""),
                    )
                    self._spot_place_tp_order(self.active_positions[-1], current_price)
                    self.pending_entry = None
                    self.last_status = f"Spot maker entry filled @ {fill_price:.5f}"
                elif age < float(getattr(self, "pending_entry_ttl_seconds", 20) or 20):
                    self.last_status = f"Waiting spot entry fill @ ${float(pending_entry.get('price', current_price) or current_price):.5f}"
                    return
                else:
                    if order_id:
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                        except Exception:
                            pass
                    self.pending_entry = None
                    self.last_status = "Spot maker entry expired"
                    return
            except Exception as e:
                logger.debug(f"Spot pending entry check skipped: {e}")

        remaining = []
        for pos in self.active_positions:
            entry = float(pos.get("entry", current_price) or current_price)
            amount = float(pos.get("amount", 0.0) or 0.0)
            support = float(pos.get("structure_support", 0.0) or 0.0)
            resistance = float(pos.get("structure_resistance", 0.0) or 0.0)
            if amount <= 0:
                continue

            tp_order_id = str(pos.get("tp_order_id") or "")
            tp_price = float(pos.get("tp_price", 0.0) or 0.0)
            emergency_price = self._spot_emergency_price(support)
            if emergency_price > 0 and current_price <= emergency_price:
                if self._spot_close_position_emergency(pos, current_price, "EMERGENCY_BREAK"):
                    continue
                remaining.append(pos)
                continue

            if tp_order_id:
                try:
                    order = self.exchange.fetch_order(tp_order_id, self.symbol)
                    status = str(order.get("status", "") or "").lower()
                    filled = float(order.get("filled", 0.0) or 0.0)
                    if status == "closed" or filled >= amount * 0.999:
                        fill_price = _order_fill_price(order, tp_price or current_price)
                        pnl = (fill_price - entry) * amount
                        profit_pct = (fill_price - entry) / entry if entry else 0.0
                        fees = (amount * entry * self.fee_rate) + (amount * fill_price * self.fee_rate)
                        exit_fee = amount * fill_price * self.fee_rate
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type("TAKE_PROFIT", net_pnl)
                        self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
                        self._log_trade(pos.get("trade_id", 0), "EXIT", "SELL", fill_price, amount, pnl, exit_fee, t_type=exit_type)
                        self.trade_count += 1
                        self._last_trade_ts = now
                        self._cleanup_trade_orders(self.symbol, pos)
                        self.last_status = f"SPOT {exit_type}: LONG @ ${fill_price:.5f} Net ${net_pnl:+,.2f}"
                        continue
                    if status == "canceled":
                        pos.pop("tp_order_id", None)
                except Exception as e:
                    logger.debug(f"Spot TP sync skipped: {e}")
                    remaining.append(pos)
                    continue

            if self._spot_update_trailing_stop(pos, current_price):
                if self._spot_close_position_emergency(pos, current_price, "TRAIL_WIN"):
                    continue
                remaining.append(pos)
                continue

            if not tp_order_id:
                self._spot_place_tp_order(pos, current_price)
            remaining.append(pos)
        self.active_positions = remaining

    def place_limit_order(self, signal: dict, symbol: str, current_price: float):
        self.symbol = _normalize_spot_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        action = str(signal.get('action', 'HOLD') or 'HOLD').upper()
        if action == "HOLD":
            self.last_status = "Signal HOLD"
            return

        now = time.time()
        if now < float(getattr(self, "_entry_block_until_ts", 0.0) or 0.0):
            wait_s = int(max(1.0, float(getattr(self, "_entry_block_until_ts", 0.0) - now)))
            self.last_status = f"Spot entry backoff ({wait_s}s)"
            return
        if self.min_seconds_between_trades and (now - float(self._last_trade_ts)) < float(self.min_seconds_between_trades):
            self.last_status = f"Cooldown ({int(self.min_seconds_between_trades)}s)"
            return
        support = self._spot_support_price(signal)
        resistance = self._spot_resistance_price(signal)
        active_layers = self._spot_layer_count()
        max_layers = max(
            1,
            int(
                getattr(
                    self,
                    "spot_max_layers",
                    getattr(self, "max_spot_layers", 3),
                )
                or 3
            ),
        )

        if action == "SELL":
            self.last_status = "Spot sell signal ignored; maker TP handles exits"
            return

        if active_layers >= max_layers and not getattr(self, "pending_entry", None):
            self.last_status = f"Spot max layers reached ({active_layers}/{max_layers})"
            return
        if self.spot_mode == "grid" and not self._spot_layer_eligible(current_price, support):
            self.last_status = "Spot layer blocked: not near support"
            return
        if getattr(self, "pending_entry", None):
            self.last_status = "Waiting spot entry fill"
            return

        quote_free = self._fetch_free_usdt()
        equity = self.get_portfolio_value(current_price)
        if quote_free <= 0:
            self.last_status = "No free spot quote balance"
            return

        configured_trade_usdt = float(getattr(self, 'fixed_trade_usdt', 0.0) or 0.0)
        reserve_pct = float(getattr(self, "spot_reserve_pct", 0.30) or 0.30)
        layer_pct = float(getattr(self, "layer_quote_pct", getattr(self, 'spot_balance_pct', 0.20)) or 0.20)
        usable_quote = max(0.0, quote_free * (1.0 - max(0.0, min(0.95, reserve_pct))))
        trade_usdt = min(quote_free * max(0.01, min(0.95, layer_pct)), usable_quote)
        if configured_trade_usdt > 0:
            trade_usdt = min(trade_usdt, configured_trade_usdt)
        if trade_usdt < 5.0:
            self.last_status = f"Spot equity too low: ${quote_free:,.2f}"
            return

        amount = trade_usdt / float(current_price)
        trade_id = self._next_trade_id
        self._next_trade_id += 1
        try:
            use_limit = bool(getattr(self, 'use_limit_orders', False))
            if use_limit:
                limit_fallback = support if support > 0 else float(signal.get('entry', current_price) or current_price)
                limit_price = _maker_entry_price(self.exchange, self.symbol, "BUY", limit_fallback)
                amount_s = self.exchange.amount_to_precision(self.symbol, amount)
                price_s = self.exchange.price_to_precision(self.symbol, limit_price)
                order_resp = self.exchange.create_order(
                    symbol=self.symbol,
                    type='LIMIT_MAKER',
                    side='BUY',
                    amount=float(amount_s),
                    price=float(price_s),
                    params={'postOnly': True}
                )
                self.pending_entry = {
                    'order_id': str(order_resp.get('id') or (order_resp.get('info', {}) or {}).get('orderId') or ''),
                    'action': "BUY",
                    'trade_id': trade_id,
                    'price': float(price_s),
                    'amount': float(amount_s),
                    'trade_usdt': float(trade_usdt),
                    'effective_leverage': 1.0,
                    'ts': now,
                    'sl': _safe_initial_sl_price("LONG", float(price_s), signal.get('sl'), self.default_sl_pct),
                    'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                    'score': float(signal.get("score", 0.0) or 0.0),
                    'confidence': float(signal.get("confidence", 0.0) or 0.0),
                    'reason': str(signal.get("reason", "") or ""),
                    'structure_support': support,
                    'structure_resistance': resistance,
                    'tp_target': float(signal.get("tp_target", 0.0) or 0.0),
                }
                self.last_status = f"Spot maker entry placed @ {price_s}"
                return

            order_resp = self.exchange.create_market_order(self.symbol, "BUY", amount)
            fill_price = _order_fill_price(order_resp, current_price)
        except Exception as e:
            logger.error(f"Spot entry error: {e}")
            err_text = str(e).lower()
            self.last_status = f"Spot entry error: {str(e)[:40]}"
            if "insufficient" in err_text:
                self._entry_block_until_ts = time.time() + 30.0
                self.last_status = "Spot entry blocked: insufficient balance (30s)"
            elif "post only" in err_text or "-5022" in err_text:
                self._entry_block_until_ts = time.time() + 10.0
                self.last_status = "Spot entry blocked: post-only reject (10s)"
            return

        sl_price = _safe_initial_sl_price("LONG", fill_price, signal.get('sl'), self.default_sl_pct)
        tp_price = self._spot_tp_price(fill_price, resistance)
        self.active_positions.append({
            'trade_id': trade_id,
            'side': "LONG",
            'entry': fill_price,
            'amount': amount,
            'trade_usdt': float(trade_usdt),
            'effective_leverage': 1.0,
            'entry_ts': now,
            'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
            'highest_price': fill_price,
            'lowest_price': 0,
            'highest_profit_pct': 0.0,
            'lowest_profit_pct': 0.0,
            'sl_pct_dist': abs(fill_price - sl_price) / fill_price if fill_price else self.default_sl_pct,
            'fee_rate': float(self.fee_rate),
                'min_profit_after_fees': float(self.min_profit_after_fees),
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
            'tp_order_id': "",
            'structure_support': signal.get('structure_support'),
            'structure_resistance': signal.get('structure_resistance'),
        })
        self.trade_count += 1
        self._last_trade_ts = now
        entry_fee = amount * fill_price * self.fee_rate
        self._log_trade(trade_id, "ENTRY", "BUY", fill_price, amount, fees=entry_fee, score=float(signal.get("score", 0.0) or 0.0), confidence=float(signal.get("confidence", 0.0) or 0.0), reason=str(signal.get("reason", "") or ""))
        self.last_status = f"SPOT BUY {amount:.6f} @ {fill_price:.5f}"

    def close_all_positions(self, symbol: str):
        target_symbol = _normalize_spot_symbol(symbol or self.symbol)
        print(f"\n[SHUTDOWN] Spot cleanup for {target_symbol}...")
        try:
            self.exchange.cancel_all_orders(target_symbol)
        except Exception as e:
            print(f"  ! Spot cancel error: {e}")
        for pos in list(self.active_positions):
            try:
                amount = float(pos.get("amount", 0.0) or 0.0)
                if amount > 0:
                    self.exchange.create_market_order(target_symbol, "SELL", amount)
                    print(f"  + Sold tracked spot position: {amount}")
            except Exception as e:
                print(f"  ! Spot close error: {e}")
        self.active_positions = []
        self.pending_entry = None
        self.pending_exit = None
        print("[SHUTDOWN] SPOT CLEANUP COMPLETE.")
