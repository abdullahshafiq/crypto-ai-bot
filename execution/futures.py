import ccxt
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
    _exchange_flag_true,
    _market_id_from_symbol,
    _next_trade_id_from_log,
    _normalize_futures_symbol,
    _order_fill_price,
    _order_id,
    _order_trigger_price,
    _order_type,
    _quote_asset_from_symbol,
    _maker_entry_price,
    _realized_exit_type,
    _runner_emergency_tp_price,
    _safe_initial_sl_price,
    _safe_tp_price,
    _trailing_tp_hit,
    _default_sl_price,
    _smart_sl_price,
)

logger = logging.getLogger(__name__)


class BinanceFuturesExecution:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "AVAX/USDC:USDC",
        leverage: int = 5,
        max_closed_trades: int = 5000,
        is_demo: bool = True,
    ):
        self.symbol = symbol
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self.is_demo = is_demo
        self.label = "BINANCE FUTURES DEMO" if is_demo else "BINANCE FUTURES LIVE"
        self.fee_rate = 0.0004
        self.fee_slippage_buffer_pct = 0.0
        self.fee_edge_multiplier = 1.0
        self.fixed_trade_usdt = 0.0
        self.learning_risk_multiplier = 1.0
        self.min_seconds_between_trades = 0
        self.min_seconds_before_reversal = 0
        self.reversal_min_confidence = 0.0
        self.reversal_min_score = 0.0
        self.reversal_min_net_edge_pct = 0.0
        self.break_even_trigger_pct = 0.0008
        self.break_even_buffer_pct = 0.0002
        self.profit_trailing_enabled = True
        self.profit_trailing_activation_pct = self.break_even_trigger_pct
        self.trailing_tp_enabled = True
        self.trailing_tp_giveback_pct = 0.08
        self.trailing_tp_min_peak_pct = 0.0015
        self.trail_tighten_1_pct = 0.0020
        self.trail_tighten_2_pct = 0.0040
        self.trail_t1_gap_pct = 0.0010
        self.trail_t2_gap_pct = 0.0008
        self.tp_pct = 0.0075
        self.default_sl_pct = 0.0018
        self.exit_on_reversal_only_in_profit = True
        self._last_trade_ts = 0.0
        self._last_profitable_exit_side = ""
        self._last_profitable_exit_ts = 0.0
        self._opposite_reset_seen_after_profit = False
        self.same_side_reentry_cooldown_seconds = 120
        self.same_side_reentry_strong_confidence = 0.85
        self.trade_log_file = "trade_log_futures.csv"
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'defaultType': 'future', # USDS-M Futures
                'adjustForTimeDifference': True,
            }
        })
        if self.is_demo:
            # Traditional Testnet endpoints (matched to keys at https://testnet.binance.vision/)
            testnet_base = 'https://testnet.binancefuture.com'
            for k, url in list(self.exchange.urls.get('api', {}).items()):
                if not isinstance(url, str) or not k.startswith('fapi'):
                    continue
                # Swap the base URL for all futures (fapi) endpoints
                self.exchange.urls['api'][k] = url.replace('https://fapi.binance.com', testnet_base)

            logger.info("Binance Futures: Running in TRADITIONAL TESTNET mode")
        else:
            logger.info("Binance Futures: Running in LIVE mode")

        # self.symbol and self.symbol_id are now set at the top of __init__
        self.leverage = leverage

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
        self._last_closed_trade_id = 0
        self._last_closed_trade_ts = 0.0
        self.pending_entry = None
        self.pending_exit = None
        self.pending_entry_ttl_seconds = 20
        self.resting_entry_ttl_seconds = 120
        self.pending_exit_ttl_seconds = 20
        self._entry_block_until_ts = 0.0

        self.max_open_positions = 1
        self.min_balance_floor = 0.0
        self.daily_loss_cap_pct = None
        self.disable_loss_cap = False
        self.dynamic_leverage_enabled = False
        self.leverage_min = 1.0
        self.leverage_max = float(leverage)
        self.leverage_confidence_levels = {}
        self.leverage_use_score = False
        self.atr_volatility_scaling = False
        self.atr_reference_pct = 0.5
        self.atr_min_multiplier = 0.3

        self.leverage_score_weight = 0.3
        self.min_profit_after_fees = 0.0002
        self.exit_on_reversal_only_in_profit = True
        self.use_native_trailing_stop = False
        self.use_exchange_stop_loss = True
        self.use_exchange_take_profit = True
        self.market_fallback_on_timeout = False
        self.trailing_stop_callback = 0.0025
        self.tp_pct = 0.0075
        self.default_sl_pct = 0.0018
        self._last_position_sync_ok = False
        self._last_flat_order_cleanup_ts = 0.0
        self._entry_block_until_ts = 0.0
        self._post_close_cleanup_needed = False
        self.scalp_config = {
            'runner_enabled': True,
            'tp_pct': 0.0075,
            'min_hold_seconds': 30,
            'runner_pullback_pct': 0.0020,
            'runner_min_lock_pct': 0.0040,
            'runner_exchange_tp_multiplier': 3.0,
            'runner_partial_exit_pct': 0.50,
            'fade_trigger_pct': 0.0020,
            'fade_exit_pct': 0.0010
        }

        self.last_status = "INIT"
        self._last_price = None

        self.initial_balance = 0.0
        self._last_equity_value = 0.0
        self._initial_price_set = False
        self.session_start = time.time()

        self._current_atr_pct = 0.02  # Track current ATR for volatility scaling

        # Set leverage and margin mode
        try:
            self.exchange.fapiPrivatePostLeverage({
                'symbol': self.symbol_id,
                'leverage': self.leverage
            })
            logger.info(f"Binance Futures: Leverage set to {self.leverage}x")
        except Exception as e:
            logger.warning(f"Leverage setup note: {e}")
            self.last_status = f"Leverage setup failed: {e}"

        # Set isolated margin mode (per-position safety, not cross-wallet risk)
        try:
            self.exchange.fapiPrivatePostMarginType({
                'symbol': self.symbol_id,
                'marginType': 'ISOLATED'
            })
            logger.info(f"Binance Futures: Margin mode set to ISOLATED (per-position safety)")
        except Exception as e:
            logger.warning(f"Isolated margin setup note: {e}")

        self._init_trade_log()
        logger.info("Binance Futures connection initialized.")

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

        risk_multiplier = float(getattr(self, 'learning_risk_multiplier', 1.0) or 1.0)
        leverage = leverage * max(0.5, min(1.0, risk_multiplier))

        # Clamp to min/max
        leverage = max(self.leverage_min, min(self.leverage_max, leverage))
        return leverage

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

    def _fetch_free_usdt(self):
        try:
            balance = self.exchange.fetch_balance()
            target_asset = _quote_asset_from_symbol(getattr(self, 'symbol', 'DOGE/USDT'))
            # 1. Standard CCXT top-level currency bucket.
            bucket = balance.get(target_asset)
            if isinstance(bucket, dict):
                free = float(bucket.get('free', 0.0) or 0.0)
                if free > 0:
                    return free
                total = float(bucket.get('total', 0.0) or 0.0)
                if total > 0:
                    return total

            # 2. Standard CCXT aggregate maps.
            free = float(balance.get('free', {}).get(target_asset, 0.0))
            if free > 0: return free

            # 3. Try total
            total = float(balance.get('total', {}).get(target_asset, 0.0))
            if total > 0: return total

            # 4. Deep dive into raw 'info' from Binance API
            if 'info' in balance:
                info = balance['info']
                # Search for any field that looks like a balance
                for key in ['availableBalance', 'totalWalletBalance', 'balance', 'withdrawAvailable']:
                    if key in info and float(info[key]) > 0:
                        return float(info[key])

                # Check nested assets list
                if 'assets' in info:
                    for asset in info['assets']:
                        if asset['asset'] == target_asset:
                            for key in ('availableBalance', 'walletBalance', 'marginBalance', 'crossWalletBalance'):
                                try:
                                    value = float(asset.get(key, 0.0) or 0.0)
                                except (TypeError, ValueError):
                                    value = 0.0
                                if value > 0:
                                    return value
            return 0.0
        except Exception as e:
            logger.error(f"Balance Fetch Error: {e}")
            self.last_status = f"Balance error: {e}"
            return 0.0

    def _fetch_free_btc(self):
        return 0.0

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

    def _ensure_native_trailing_stop(self, pos: dict) -> bool:
        """
        Place a real exchange trailing stop for an active position once.
        """
        if not isinstance(pos, dict):
            return False
        if not bool(pos.get("trail_armed", False)):
            existing_id = str(pos.get('native_trailing_order_id') or '')
            if existing_id:
                try:
                    self.exchange.cancel_order(existing_id, self.symbol)
                except Exception:
                    pass
                pos.pop('native_trailing_order_id', None)
                pos.pop('native_trailing_activation_price', None)
                pos.pop('native_trailing_callback_pct', None)
            return False
        if not getattr(self, 'use_native_trailing_stop', False):
            existing_id = str(pos.get('native_trailing_order_id') or '') if isinstance(pos, dict) else ''
            if existing_id:
                try:
                    self.exchange.cancel_order(existing_id, self.symbol)
                except Exception:
                    pass
                if isinstance(pos, dict):
                    pos.pop('native_trailing_order_id', None)
                    pos.pop('native_trailing_activation_price', None)
                    pos.pop('native_trailing_callback_pct', None)
            return False

        side = str(pos.get('side', '')).upper()
        amount = float(pos.get('amount', 0.0) or 0.0)
        entry_price = float(pos.get('entry', self._last_price or 0.0) or self._last_price or 0.0)
        if side not in {'LONG', 'SHORT'} or amount <= 0 or entry_price <= 0:
            return False

        order_side = 'SELL' if side == 'LONG' else 'BUY'
        callback_rate_pct = float(getattr(self, 'trailing_stop_callback', 0.005)) * 100
        activation_pct = float(pos.get('native_trailing_activation_pct', pos.get('break_even_trigger_pct', 0.0020)) or 0.0020)
        activation_pct = max(0.0020, activation_pct)
        activation_price = entry_price * (1 + activation_pct) if side == 'LONG' else entry_price * (1 - activation_pct)

        existing_id = str(pos.get('native_trailing_order_id') or '')
        existing_activation = float(pos.get('native_trailing_activation_price', 0.0) or 0.0)
        matching_orders = self._matching_reduce_only_orders(order_side, {"TRAILING_STOP_MARKET"})
        if existing_id and existing_activation > 0:
            if abs(existing_activation - activation_price) / activation_price > 0.0005:
                try:
                    self.exchange.cancel_order(existing_id, self.symbol)
                except Exception:
                    pass
                pos.pop('native_trailing_order_id', None)
                pos.pop('native_trailing_activation_price', None)
                pos.pop('native_trailing_callback_pct', None)
            else:
                existing_live = any(_order_id(order) == existing_id for order in matching_orders)
                if existing_live:
                    self._cancel_reduce_only_orders(order_side, {"TRAILING_STOP_MARKET"}, keep_id=existing_id)
                    return False
                logger.warning(
                    f"[EXCHANGE] Stored trailing stop order {existing_id[-8:]} is not open; recreating protection."
                )
                pos.pop('native_trailing_order_id', None)
                pos.pop('native_trailing_activation_price', None)
                pos.pop('native_trailing_callback_pct', None)

        if matching_orders:
            exact_match = None
            for order in matching_orders:
                trigger_price = _order_trigger_price(order)
                if trigger_price > 0 and abs(trigger_price - activation_price) / activation_price <= 0.0005:
                    exact_match = order
                    break
            if exact_match:
                keep_id = _order_id(exact_match)
                if keep_id:
                    pos['native_trailing_order_id'] = keep_id
                    pos['native_trailing_activation_price'] = float(activation_price)
                    pos['native_trailing_callback_pct'] = callback_rate_pct
                self._cancel_reduce_only_orders(order_side, {"TRAILING_STOP_MARKET"}, keep_id=keep_id)
                return False
            self._cancel_reduce_only_orders(order_side, {"TRAILING_STOP_MARKET"})
            if self._matching_reduce_only_orders(order_side, {"TRAILING_STOP_MARKET"}):
                logger.debug("Trailing stop replacement waiting for existing trailing orders to clear")
                return False

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='TRAILING_STOP_MARKET',
                side=order_side,
                amount=float(amount),
                params={
                    'callbackRate': callback_rate_pct,
                    'activationPrice': self.exchange.price_to_precision(self.symbol, activation_price),
                    'reduceOnly': True,
                    'workingType': 'MARK_PRICE'
                }
            )
            pos['native_trailing_order_id'] = str(order.get('id') or (order.get('info', {}) or {}).get('orderId') or '')
            pos['native_trailing_callback_pct'] = callback_rate_pct
            pos['native_trailing_activation_price'] = float(activation_price)
            logger.info(f"[EXCHANGE] Native Trailing Stop set at {callback_rate_pct}% (Activates at {activation_price})")
            return True
        except Exception as e:
            logger.warning(f"Failed to place Native Trailing Stop: {e}")
            return False

    def _cancel_reduce_only_orders(self, order_side: str, order_types: set[str], keep_id: str = "") -> int:
        """
        Cancel orphan exchange-side protection orders by type/side.
        Binance/CCXT can fail to round-trip a conditional order id consistently, so
        replacement must also clean up matching reduce-only open orders from the book.
        """
        cancelled = 0
        side_u = str(order_side or "").upper()
        keep_id = str(keep_id or "")
        types = {str(t or "").upper().replace(" ", "_") for t in (order_types or set())}
        orders = self._fetch_open_protection_orders(self.symbol)

        for order in orders:
            info = order.get("info", {}) or {}
            reduce_only = _exchange_flag_true(order.get("reduceOnly")) or _exchange_flag_true(info.get("reduceOnly"))
            if not reduce_only:
                continue
            if str(order.get("side") or info.get("side") or "").upper() != side_u:
                continue
            order_type = _order_type(order)
            if order_type not in types:
                continue
            order_id = _order_id(order)
            if not order_id or (keep_id and order_id == keep_id):
                continue
            try:
                self.exchange.cancel_order(order_id, self.symbol)
                cancelled += 1
            except Exception as e:
                logger.debug(f"Protection order cancel skipped ({order_type} {order_id}): {e}")
        return cancelled

    def _reset_excess_protection_orders(self, pos: dict, threshold: int = 3) -> bool:
        """
        Hard reset protection orders if the stack has clearly drifted out of sync.
        This prevents repeated stop/TP refreshes from leaving many orphan orders behind.
        """
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        if side not in {"LONG", "SHORT"}:
            return False

        order_side = "SELL" if side == "LONG" else "BUY"
        orders = self._fetch_open_protection_orders(self.symbol)
        protect_types = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"}
        protect_orders = []
        for order in orders:
            info = order.get("info", {}) or {}
            reduce_only = _exchange_flag_true(order.get("reduceOnly")) or _exchange_flag_true(info.get("reduceOnly"))
            if not reduce_only:
                continue
            if str(order.get("side") or info.get("side") or "").upper() != order_side:
                continue
            if _order_type(order) not in protect_types:
                continue
            protect_orders.append(order)

        if len(protect_orders) <= int(threshold or 0):
            return False

        logger.warning(
            f"[CLEANUP] Protection stack bloated for {self.symbol} ({len(protect_orders)} orders). "
            "Resetting stop/TP/trailing orders."
        )

        for order in protect_orders:
            order_id = _order_id(order)
            if not order_id:
                continue
            try:
                self.exchange.cancel_order(order_id, self.symbol)
            except Exception as e:
                logger.debug(f"Excess protection cancel skipped ({order_id}): {e}")

        pos.pop("exchange_stop_order_id", None)
        pos.pop("exchange_stop_price", None)
        pos.pop("exchange_stop_order_ts", None)
        pos.pop("exchange_tp_order_id", None)
        pos.pop("exchange_tp_price", None)
        pos.pop("exchange_tp_order_ts", None)
        pos.pop("native_trailing_order_id", None)
        pos.pop("native_trailing_activation_price", None)
        pos.pop("native_trailing_callback_pct", None)

        return True

    def _fetch_open_protection_orders(self, symbol: str = None) -> list[dict]:
        """
        Fetch open orders through both CCXT and Binance raw futures API.
        Conditional TP/SL orders can be missing or delayed in one path.
        """
        target_symbol = symbol or self.symbol
        target_id = _market_id_from_symbol(target_symbol).upper()
        orders = []
        seen = set()

        def add_order(order: dict):
            if not isinstance(order, dict):
                return
            oid = _order_id(order)
            key = oid or str(order)
            if key in seen:
                return
            seen.add(key)
            orders.append(order)

        try:
            for order in self.exchange.fetch_open_orders(target_symbol) or []:
                add_order(order)
        except Exception as e:
            logger.debug(f"Protection CCXT order fetch skipped: {e}")

        try:
            for raw in self.exchange.fapiPrivateGetOpenOrders({"symbol": target_id}) or []:
                add_order(raw)
        except Exception as e:
            logger.debug(f"Protection raw order fetch skipped: {e}")

        return orders

    def _wipe_all_orphans(self, symbol: str):
        """
        Nuclear cleanup: Fetch ALL open orders for the symbol and cancel any that are not
        explicitly tracked in our local state.
        """
        target_symbol = symbol or self.symbol
        target_id = _market_id_from_symbol(target_symbol).upper()

        # 1. CCXT Bulk Cancel (Limit Orders)
        try:
            self.exchange.cancel_all_orders(target_symbol)
        except Exception:
            pass

        # 2. Binance Direct Bulk Cancel (Conditional Orders)
        try:
            self.exchange.fapiPrivateDeleteAllOpenOrders({'symbol': target_id})
        except Exception:
            pass

        # 3. Individual Sweep Fallback (Catch anything that survived bulk)
        try:
            open_orders = self.exchange.fetch_open_orders(target_symbol)
            if open_orders:
                tracked_ids = set()
                if self.active_positions:
                    pos = self.active_positions[0]
                    for key in ['exchange_tp_order_id', 'exchange_stop_order_id', 'native_trailing_order_id']:
                        val = str(pos.get(key) or "")
                        if val: tracked_ids.add(val)

                if getattr(self, "pending_entry", None):
                    val = str(self.pending_entry.get("order_id") or "")
                    if val: tracked_ids.add(val)

                if getattr(self, "pending_exit", None):
                    val = str(self.pending_exit.get("order_id") or "")
                    if val: tracked_ids.add(val)

                cancelled = 0
                for o in open_orders:
                    order_id = _order_id(o)
                    if order_id and order_id not in tracked_ids:
                        try:
                            self.exchange.cancel_order(order_id, target_symbol)
                            cancelled += 1
                        except Exception:
                            pass
                if cancelled:
                    logger.info(f"[CLEANUP] Ruthlessly killed {cancelled} orphan order(s) on {target_symbol}")
        except Exception as e:
            logger.debug(f"Orphan sweep failed: {e}")

    def _matching_reduce_only_orders(self, order_side: str, order_types: set[str]) -> list[dict]:
        side_u = str(order_side or "").upper()
        types = {str(t or "").upper().replace(" ", "_") for t in (order_types or set())}
        orders = self._fetch_open_protection_orders(self.symbol)

        matching = []
        for order in orders:
            info = order.get("info", {}) or {}
            reduce_only = _exchange_flag_true(order.get("reduceOnly")) or _exchange_flag_true(info.get("reduceOnly"))
            if not reduce_only:
                continue
            if str(order.get("side") or info.get("side") or "").upper() != side_u:
                continue
            if _order_type(order) not in types:
                continue
            matching.append(order)
        return matching

    def _ensure_exchange_stop_loss(self, pos: dict) -> bool:
        """
        Maintain a reduce-only STOP_MARKET order at the current local stop.
        This is the hard exchange-side backstop for process/network failure.
        """
        if not getattr(self, "use_exchange_stop_loss", True):
            return False
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        amount = float(pos.get("amount", 0.0) or 0.0)
        stop_price = float(pos.get("sl", 0.0) or 0.0)
        current_price = float(getattr(self, "_last_price", 0.0) or 0.0)
        if side not in {"LONG", "SHORT"} or amount <= 0 or stop_price <= 0 or current_price <= 0:
            return False

        if side == "LONG" and stop_price >= current_price:
            return False
        if side == "SHORT" and stop_price <= current_price:
            return False

        order_side = "SELL" if side == "LONG" else "BUY"
        existing_id = str(pos.get("exchange_stop_order_id") or "")
        existing_stop = float(pos.get("exchange_stop_price", 0.0) or 0.0)
        recent_ts = float(pos.get("exchange_stop_order_ts", 0.0) or 0.0)
        # Short-circuit: skip API fetch if price matches and we confirmed/placed within 60s
        if existing_id and existing_stop > 0 and abs(existing_stop - stop_price) / stop_price <= 0.0005:
            if recent_ts > 0 and time.time() - recent_ts < 60.0:
                return False
        matching_orders = self._matching_reduce_only_orders(order_side, {"STOP_MARKET"})
        if existing_id and existing_stop > 0:
            if abs(existing_stop - stop_price) / stop_price <= 0.0005:
                existing_live = any(_order_id(order) == existing_id for order in matching_orders)
                if existing_live:
                    # If we have a local ID and it matches the price, we're good.
                    # But also wipe any OTHER stop orders that might be orphans.
                    self._cancel_reduce_only_orders(order_side, {"STOP_MARKET"}, keep_id=existing_id)
                    pos["exchange_stop_order_id"] = existing_id
                    pos["exchange_stop_price"] = float(stop_price)
                    pos["exchange_stop_order_ts"] = time.time()  # refresh so short-circuit holds
                    return False
                if recent_ts > 0 and time.time() - recent_ts < 60:
                    logger.debug(f"[EXCHANGE] Waiting for SL order {existing_id[-8:]} to appear before recreating.")
                    return False
                logger.warning(
                    f"[EXCHANGE] Stored stop-loss order {existing_id[-8:]} is not open; recreating protection."
                )
                pos.pop("exchange_stop_order_id", None)
                pos.pop("exchange_stop_price", None)
                existing_id = ""
                existing_stop = 0.0
            if existing_id:
                try:
                    self.exchange.cancel_order(existing_id, self.symbol)
                except Exception as e:
                    logger.debug(f"Stop-loss replace cancel skipped: {e}")
            pos.pop("exchange_stop_order_id", None)
            pos.pop("exchange_stop_price", None)

        # NEW: Adoption Logic - if we don't have a local ID, check if an order already exists on exchange
        if matching_orders:
            exact_match = None
            for order in matching_orders:
                trigger_price = _order_trigger_price(order)
                # If an order exists with the same price, adopt it!
                if trigger_price > 0 and abs(trigger_price - stop_price) / stop_price <= 0.0005:
                    exact_match = order
                    break

            if exact_match:
                keep_id = _order_id(exact_match)
                if keep_id:
                    pos["exchange_stop_order_id"] = keep_id
                    pos["exchange_stop_price"] = float(stop_price)
                    logger.info(f"[EXCHANGE] Adopted existing SL order {keep_id[-8:]} @ {stop_price:.5f}")
                # Wipe any duplicates that are NOT the one we adopted
                self._cancel_reduce_only_orders(order_side, {"STOP_MARKET"}, keep_id=keep_id)
                return False
            else:
                # If they don't match our price, kill them all before placing new one
                cancelled = self._cancel_reduce_only_orders(order_side, {"STOP_MARKET"})
                if cancelled:
                    logger.info(f"[EXCHANGE] Purged {cancelled} non-matching SL orphans.")

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="STOP_MARKET",
                side=order_side,
                amount=float(self.exchange.amount_to_precision(self.symbol, amount)),
                price=None,
                params={
                    "stopPrice": self.exchange.price_to_precision(self.symbol, stop_price),
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            pos["exchange_stop_order_id"] = _order_id(order)
            pos["exchange_stop_price"] = float(stop_price)
            pos["exchange_stop_order_ts"] = time.time()
            logger.info(f"[EXCHANGE] Hard stop-loss set at {stop_price:.5f}")
            return True
        except Exception as e:
            logger.warning(f"Failed to place hard stop-loss: {e}")
            return False

    def _ensure_exchange_take_profit(self, pos: dict) -> bool:
        """
        Maintain a reduce-only TAKE_PROFIT_MARKET order at the active TP.
        This lets Binance react to the target even if the bot loop is delayed.
        """
        if not getattr(self, "use_exchange_take_profit", True):
            return False
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        amount = float(pos.get("amount", 0.0) or 0.0)
        tp_price = float(pos.get("tp_price", 0.0) or 0.0)
        current_price = float(getattr(self, "_last_price", 0.0) or 0.0)
        if side not in {"LONG", "SHORT"} or amount <= 0 or tp_price <= 0 or current_price <= 0:
            return False

        if side == "LONG" and tp_price <= current_price:
            return False
        if side == "SHORT" and tp_price >= current_price:
            return False

        order_side = "SELL" if side == "LONG" else "BUY"
        existing_id = str(pos.get("exchange_tp_order_id") or "")
        existing_tp = float(pos.get("exchange_tp_price", 0.0) or 0.0)
        recent_ts = float(pos.get("exchange_tp_order_ts", 0.0) or 0.0)
        # Short-circuit: skip API fetch if price matches and we confirmed/placed within 60s
        if existing_id and existing_tp > 0 and abs(existing_tp - tp_price) / tp_price <= 0.0005:
            if recent_ts > 0 and time.time() - recent_ts < 60.0:
                return False
        matching_orders = self._matching_reduce_only_orders(order_side, {"TAKE_PROFIT_MARKET"})
        if existing_id and existing_tp > 0:
            if abs(existing_tp - tp_price) / tp_price <= 0.0005:
                existing_live = any(_order_id(order) == existing_id for order in matching_orders)
                if existing_live:
                    # Matching local order found. Keep it, but kill orphans.
                    self._cancel_reduce_only_orders(order_side, {"TAKE_PROFIT_MARKET"}, keep_id=existing_id)
                    pos["exchange_tp_order_id"] = existing_id
                    pos["exchange_tp_price"] = float(tp_price)
                    pos["exchange_tp_order_ts"] = time.time()  # refresh so short-circuit holds
                    return False
                if recent_ts > 0 and time.time() - recent_ts < 60:
                    logger.debug(f"[EXCHANGE] Waiting for TP order {existing_id[-8:]} to appear before recreating.")
                    return False
                logger.warning(
                    f"[EXCHANGE] Stored take-profit order {existing_id[-8:]} is not open; recreating protection."
                )
                pos.pop("exchange_tp_order_id", None)
                pos.pop("exchange_tp_price", None)
                existing_id = ""
                existing_tp = 0.0
            if existing_id:
                try:
                    self.exchange.cancel_order(existing_id, self.symbol)
                except Exception as e:
                    logger.debug(f"Take-profit replace cancel skipped: {e}")
            pos.pop("exchange_tp_order_id", None)
            pos.pop("exchange_tp_price", None)

        # NEW: Adoption Logic for Take Profit
        if matching_orders:
            exact_match = None
            for order in matching_orders:
                trigger_price = _order_trigger_price(order)
                if trigger_price > 0 and abs(trigger_price - tp_price) / tp_price <= 0.0005:
                    exact_match = order
                    break

            if exact_match:
                keep_id = _order_id(exact_match)
                if keep_id:
                    pos["exchange_tp_order_id"] = keep_id
                    pos["exchange_tp_price"] = float(tp_price)
                    logger.info(f"[EXCHANGE] Adopted existing TP order {keep_id[-8:]} @ {tp_price:.5f}")
                self._cancel_reduce_only_orders(order_side, {"TAKE_PROFIT_MARKET"}, keep_id=keep_id)
                return False
            else:
                # Non-matching orphans found. Purge them.
                cancelled = self._cancel_reduce_only_orders(order_side, {"TAKE_PROFIT_MARKET"})
                if cancelled:
                    logger.info(f"[EXCHANGE] Purged {cancelled} non-matching TP orphans.")

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="TAKE_PROFIT_MARKET",
                side=order_side,
                amount=float(self.exchange.amount_to_precision(self.symbol, amount)),
                price=None,
                params={
                    "stopPrice": self.exchange.price_to_precision(self.symbol, tp_price),
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            pos["exchange_tp_order_id"] = _order_id(order)
            pos["exchange_tp_price"] = float(tp_price)
            pos["exchange_tp_order_ts"] = time.time()
            logger.info(f"[EXCHANGE] Take-profit set at {tp_price:.5f}")
            return True
        except Exception as e:
            logger.warning(f"Failed to place take-profit: {e}")
            return False

    def _log_exchange_close_diagnostics(self, pos: dict, current_price: float) -> None:
        """
        Capture recent Binance fills/orders when a locally tracked position is gone
        from the exchange. This turns EXCHANGE_CLOSED into an auditable event.
        """
        if not isinstance(pos, dict):
            return

        try:
            entry_ts = float(pos.get("entry_ts", time.time()) or time.time())
        except (TypeError, ValueError):
            entry_ts = time.time()
        since_ms = max(0, int((entry_ts - 300) * 1000))

        side = str(pos.get("side", "") or "").upper()
        close_side = "sell" if side == "LONG" else "buy"
        expected_amount = float(pos.get("amount", 0.0) or 0.0)
        tracked_ids = {
            str(pos.get("exchange_stop_order_id") or ""),
            str(pos.get("exchange_tp_order_id") or ""),
            str(pos.get("native_trailing_order_id") or ""),
        }
        tracked_ids.discard("")

        logger.warning(
            "[SYNC_DIAG] Exchange flat for %s while local %s %.8f @ %.5f remained. "
            "Tracked reduce-only ids=%s current=%.5f",
            self.symbol,
            side or "?",
            expected_amount,
            float(pos.get("entry", 0.0) or 0.0),
            ",".join(sorted(tracked_ids)) or "-",
            float(current_price or 0.0),
        )

        try:
            trades = self.exchange.fetch_my_trades(self.symbol, since=since_ms, limit=20) or []
            recent_trades = sorted(trades, key=lambda t: int(t.get("timestamp") or 0))[-8:]
            for trade in recent_trades:
                info = trade.get("info", {}) or {}
                ts = int(trade.get("timestamp") or 0)
                ts_s = datetime.utcfromtimestamp(ts / 1000).isoformat() if ts > 0 else "-"
                logger.warning(
                    "[SYNC_DIAG] recent_fill ts=%s id=%s order=%s side=%s price=%s amount=%s cost=%s fee=%s maker=%s",
                    ts_s,
                    trade.get("id") or info.get("id") or "-",
                    trade.get("order") or info.get("orderId") or "-",
                    str(trade.get("side") or info.get("side") or "").lower() or "-",
                    trade.get("price") or info.get("price") or "-",
                    trade.get("amount") or info.get("qty") or "-",
                    trade.get("cost") or info.get("quoteQty") or "-",
                    trade.get("fee") or "-",
                    trade.get("takerOrMaker") or "-",
                )
        except Exception as e:
            logger.warning(f"[SYNC_DIAG] fetch_my_trades failed: {e}")

        for fetcher_name in ("fetch_closed_orders", "fetch_orders"):
            fetcher = getattr(self.exchange, fetcher_name, None)
            if not callable(fetcher):
                continue
            try:
                orders = fetcher(self.symbol, since=since_ms, limit=20) or []
                recent_orders = sorted(orders, key=lambda o: int(o.get("timestamp") or 0))[-8:]
                for order in recent_orders:
                    info = order.get("info", {}) or {}
                    oid = _order_id(order)
                    order_side = str(order.get("side") or info.get("side") or "").lower()
                    reduce_only = order.get("reduceOnly") if order.get("reduceOnly") is not None else info.get("reduceOnly")
                    if oid not in tracked_ids and order_side != close_side and not _exchange_flag_true(reduce_only):
                        continue
                    ts = int(order.get("timestamp") or 0)
                    ts_s = datetime.utcfromtimestamp(ts / 1000).isoformat() if ts > 0 else "-"
                    logger.warning(
                        "[SYNC_DIAG] recent_order source=%s ts=%s id=%s client=%s type=%s side=%s status=%s "
                        "filled=%s amount=%s avg=%s price=%s stop=%s reduceOnly=%s closePosition=%s",
                        fetcher_name,
                        ts_s,
                        oid or "-",
                        order.get("clientOrderId") or info.get("clientOrderId") or "-",
                        _order_type(order) or "-",
                        order_side or "-",
                        order.get("status") or info.get("status") or "-",
                        order.get("filled") or info.get("executedQty") or "-",
                        order.get("amount") or info.get("origQty") or "-",
                        order.get("average") or info.get("avgPrice") or "-",
                        order.get("price") or info.get("price") or "-",
                        _order_trigger_price(order) or "-",
                        reduce_only,
                        info.get("closePosition"),
                    )
            except Exception as e:
                logger.warning(f"[SYNC_DIAG] {fetcher_name} failed: {e}")

    def _resolve_exchange_close_fill_price(self, pos: dict, current_price: float) -> float:
        """
        Query Binance for the actual fill price when the sync path detects a position
        is gone from the exchange. Checks tracked TP/SL/trailing order IDs first, then
        falls back to recent trade fills. Returns current_price if nothing found.
        """
        if not isinstance(pos, dict):
            return current_price

        try:
            entry_ts = float(pos.get("entry_ts", time.time()) or time.time())
        except (TypeError, ValueError):
            entry_ts = time.time()
        since_ms = max(0, int((entry_ts - 60) * 1000))

        side = str(pos.get("side", "") or "").upper()
        close_side = "sell" if side == "LONG" else "buy"
        expected_amount = float(pos.get("amount", 0.0) or 0.0)

        tracked_ids = {
            str(pos.get("exchange_stop_order_id") or ""),
            str(pos.get("exchange_tp_order_id") or ""),
            str(pos.get("native_trailing_order_id") or ""),
        }
        tracked_ids.discard("")

        # Check tracked reduce-only orders for a filled average price
        for order_id in tracked_ids:
            try:
                order = self.exchange.fetch_order(order_id, self.symbol)
                status = str(order.get("status") or "").lower()
                avg = float(order.get("average") or (order.get("info", {}) or {}).get("avgPrice") or 0.0)
                filled = float(order.get("filled") or (order.get("info", {}) or {}).get("executedQty") or 0.0)
                if status in ("closed", "filled") and avg > 0 and filled > 0:
                    logger.info(
                        "[SYNC] Resolved actual fill price from order %s: %.5f (was using current=%.5f)",
                        order_id[-8:], avg, current_price,
                    )
                    return avg
            except Exception:
                pass

        # Fall back to recent trade fills matching the close side and amount
        try:
            trades = self.exchange.fetch_my_trades(self.symbol, since=since_ms, limit=20) or []
            close_trades = [
                t for t in trades
                if str(t.get("side") or "").lower() == close_side
                and float(t.get("amount") or 0.0) > 0
            ]
            if close_trades:
                # Use the most recent close fill
                close_trades.sort(key=lambda t: int(t.get("timestamp") or 0))
                best = close_trades[-1]
                fill_price = float(best.get("price") or 0.0)
                if fill_price > 0:
                    logger.info(
                        "[SYNC] Resolved actual fill price from recent trade: %.5f (was using current=%.5f)",
                        fill_price, current_price,
                    )
                    return fill_price
        except Exception as e:
            logger.debug(f"[SYNC] fetch_my_trades for fill price resolution failed: {e}")

        logger.debug("[SYNC] Could not resolve actual fill price; using current_price=%.5f", current_price)
        return current_price

    def _cleanup_trade_orders(self, symbol: str = None, pos: dict = None):
        """
        Cancel all open trade-related orders and clear local pending order state.
        """
        target_symbol = symbol or self.symbol
        target_id = _market_id_from_symbol(target_symbol).upper()
        try:
            self.exchange.cancel_all_orders(target_symbol)
        except Exception as e:
            logger.debug(f"Cleanup cancel_all_orders skipped: {e}")
        try:
            self.exchange.fapiPrivateDeleteAllOpenOrders({'symbol': target_id})
        except Exception as e:
            logger.debug(f"Cleanup direct cancel_all_open_orders skipped: {e}")

        if isinstance(pos, dict):
            trail_id = str(pos.get('native_trailing_order_id') or '')
            if trail_id:
                try:
                    self.exchange.cancel_order(trail_id, target_symbol)
                except Exception as e:
                    logger.debug(f"Cleanup cancel trailing skipped: {e}")
            stop_id = str(pos.get('exchange_stop_order_id') or '')
            if stop_id:
                try:
                    self.exchange.cancel_order(stop_id, target_symbol)
                except Exception as e:
                    logger.debug(f"Cleanup cancel stop skipped: {e}")
            tp_id = str(pos.get('exchange_tp_order_id') or '')
            if tp_id:
                try:
                    self.exchange.cancel_order(tp_id, target_symbol)
                except Exception as e:
                    logger.debug(f"Cleanup cancel take-profit skipped: {e}")
            pos.pop('native_trailing_order_id', None)
            pos.pop('native_trailing_activation_price', None)
            pos.pop('native_trailing_callback_pct', None)
            pos.pop('exchange_stop_order_id', None)
            pos.pop('exchange_stop_price', None)
            pos.pop('exchange_stop_order_ts', None)
            pos.pop('exchange_tp_order_id', None)
            pos.pop('exchange_tp_price', None)
            pos.pop('exchange_tp_order_ts', None)

        # Individual sweep fallback — cancel any orders that bulk cancel missed
        try:
            remaining = self.exchange.fetch_open_orders(target_symbol)
            for o in (remaining or []):
                rid = str(o.get('id') or (o.get('info', {}) or {}).get('orderId') or '')
                if rid:
                    try:
                        self.exchange.cancel_order(rid, target_symbol)
                    except Exception:
                        pass
            if remaining:
                logger.info(f"[ORDER] Cleanup sweep: cancelled {len(remaining)} remaining orders")
        except Exception:
            pass

        if getattr(self, "pending_entry", None):
            self.pending_entry = None
        if getattr(self, "pending_exit", None):
            self.pending_exit = None

    def _execute_futures_reduce_only_exit(self, pos: dict, current_price: float, exit_type: str, trade_id: int) -> bool:
        """
        Execute a futures close. Returns True when the position should remain open
        after the call, and False when it has been fully closed.
        """
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        amount = float(pos.get("amount", 0.0) or 0.0)
        entry = float(pos.get("entry", current_price) or current_price)
        if side not in {"LONG", "SHORT"} or amount <= 0 or entry <= 0:
            return False

        order_side = "SELL" if side == "LONG" else "BUY"
        scalp_cfg = getattr(self, "scalp_config", {}) or {}
        partial_pct = float(scalp_cfg.get("runner_partial_exit_pct", 0.0) or 0.0)
        use_scale_out = (
            str(exit_type or "").upper() == "SCALP_EXIT"
            and not bool(pos.get("runner_scale_out_taken", False))
            and 0.0 < partial_pct < 1.0
        )

        close_amount = amount * partial_pct if use_scale_out else amount
        try:
            close_amount = float(self.exchange.amount_to_precision(self.symbol, close_amount))
        except Exception:
            close_amount = float(close_amount)
        if close_amount <= 0:
            return False
        if close_amount >= amount * 0.999:
            use_scale_out = False
            close_amount = amount

        # Remove stale protection before the reduce-only close so the
        # replacement orders can be rebuilt for the remaining size.
        self._cleanup_trade_orders(self.symbol, pos)

        exit_type_u = str(exit_type or "").upper()
        profit_exit = exit_type_u in {
            "TAKE_PROFIT",
            "TRAIL_TP",
            "SCALP_EXIT",
            "SCALP_EXIT_PARTIAL",
            "SCALP_FADE",
            "TTL_EXIT",
            "TRAIL_WIN",
        }
        emergency_exit = exit_type_u in {"STOP_LOSS", "TRAIL_SL"}
        stop_like_exit = exit_type_u in {
            "STOP_LOSS",
            "TRAIL_WIN",
            "TRAIL_SL",
            "TAKE_PROFIT",
            "TRAIL_TP",
            "SCALP_EXIT",
            "SCALP_EXIT_PARTIAL",
            "SCALP_FADE",
            "TTL_EXIT",
        }
        if getattr(self, "use_limit_orders", False):
            try:
                exit_limit = _maker_entry_price(self.exchange, self.symbol, order_side, float(current_price))
                price_s = self.exchange.price_to_precision(self.symbol, exit_limit)
                maker_price = float(price_s)
                projected_profit_pct = (maker_price - entry) / entry if side == "LONG" else (entry - maker_price) / entry
                required_profit_pct = (
                    (2.0 * float(getattr(self, "fee_rate", 0.0) or 0.0))
                    + float(getattr(self, "fee_slippage_buffer_pct", 0.0) or 0.0)
                    + float(pos.get("min_profit_after_fees", getattr(self, "min_profit_after_fees", 0.0)) or 0.0)
                )
                if profit_exit and projected_profit_pct < required_profit_pct:
                    self.last_status = (
                        f"{exit_type_u}: maker exit skipped; net edge "
                        f"{projected_profit_pct:+.2%} < {required_profit_pct:.2%}"
                    )
                    logger.info(
                        "[MAKER_EXIT] Skipped %s for %s: projected edge %.4f < required %.4f",
                        exit_type_u,
                        side,
                        projected_profit_pct,
                        required_profit_pct,
                    )
                    self._ensure_exchange_stop_loss(pos)
                    self._ensure_exchange_take_profit(pos)
                    self._ensure_native_trailing_stop(pos)
                    return True
                order_resp = self.exchange.create_order(
                    symbol=self.symbol,
                    type="LIMIT",
                    side=order_side,
                    amount=close_amount,
                    price=maker_price,
                    params={"reduceOnly": True, "postOnly": True, "timeInForce": "GTX"},
                )
                ttl_seconds = 1.0 if profit_exit else (3.0 if stop_like_exit else float(getattr(self, "pending_exit_ttl_seconds", 20) or 20))
                self.pending_exit = {
                    "order_id": str(order_resp.get("id") or (order_resp.get("info", {}) or {}).get("orderId") or ""),
                    "ts": time.time(),
                    "price": maker_price,
                    "amount": close_amount,
                    "side": order_side,
                    "exit_type": exit_type,
                    "ttl_seconds": ttl_seconds,
                    "fallback_market": bool(emergency_exit or (getattr(self, "market_fallback_on_timeout", False) and not profit_exit)),
                }
                if profit_exit:
                    self._ensure_exchange_stop_loss(pos)
                    self._ensure_native_trailing_stop(pos)
                self.last_status = f"Maker exit placed @ {price_s}"
                logger.info(f"[MAKER_EXIT] {order_side} placed @ {price_s} | ttl={ttl_seconds:.1f}s")
                return True
            except Exception as e:
                logger.warning(f"Maker exit placement failed: {e}")
                if stop_like_exit or getattr(self, "market_fallback_on_timeout", False):
                    logger.warning("Maker exit placement failed; falling back to market close")
                    return self._finalize_reduce_only_market_exit(
                        pos,
                        current_price,
                        exit_type,
                        trade_id,
                        amount=close_amount,
                        use_scale_out=use_scale_out,
                    )
                self.last_status = "Maker exit failed; still tracking position"
                self._ensure_exchange_stop_loss(pos)
                self._ensure_exchange_take_profit(pos)
                self._ensure_native_trailing_stop(pos)
                return True

        return self._finalize_reduce_only_market_exit(
            pos,
            current_price,
            exit_type,
            trade_id,
            amount=close_amount,
            use_scale_out=use_scale_out,
        )

    def _finalize_reduce_only_market_exit(
        self,
        pos: dict,
        current_price: float,
        exit_type: str,
        trade_id: int,
        amount: float | None = None,
        use_scale_out: bool | None = None,
    ) -> bool:
        if not isinstance(pos, dict):
            return False

        side = str(pos.get("side", "")).upper()
        pos_amount = float(pos.get("amount", 0.0) or 0.0)
        entry = float(pos.get("entry", current_price) or current_price)
        if side not in {"LONG", "SHORT"} or pos_amount <= 0 or entry <= 0:
            return False

        order_side = "SELL" if side == "LONG" else "BUY"
        scalp_cfg = getattr(self, "scalp_config", {}) or {}
        partial_pct = float(scalp_cfg.get("runner_partial_exit_pct", 0.0) or 0.0)
        if use_scale_out is None:
            use_scale_out = (
                str(exit_type or "").upper() == "SCALP_EXIT"
                and not bool(pos.get("runner_scale_out_taken", False))
                and 0.0 < partial_pct < 1.0
            )

        close_amount = float(amount if amount is not None else (pos_amount * partial_pct if use_scale_out else pos_amount))
        try:
            close_amount = float(self.exchange.amount_to_precision(self.symbol, close_amount))
        except Exception:
            close_amount = float(close_amount)
        if close_amount <= 0:
            return False
        if close_amount >= pos_amount * 0.999:
            use_scale_out = False
            close_amount = pos_amount

        try:
            order_resp = self.exchange.create_market_order(
                self.symbol,
                order_side,
                close_amount,
                params={"reduceOnly": True},
            )
            fill_price = _order_fill_price(order_resp, current_price)
            filled_amount = close_amount
            try:
                filled_amount = float(
                    order_resp.get("filled")
                    or (order_resp.get("info", {}) or {}).get("executedQty")
                    or close_amount
                )
            except Exception:
                filled_amount = close_amount
            filled_amount = max(0.0, min(pos_amount, filled_amount))
        except Exception as e:
            logger.warning(f"Market exit failed; keeping position active: {e}")
            self.last_status = "Market exit failed; still tracking position"
            self._ensure_exchange_stop_loss(pos)
            self._ensure_exchange_take_profit(pos)
            self._ensure_native_trailing_stop(pos)
            return True

        pnl = (fill_price - entry) * filled_amount if side == "LONG" else (entry - fill_price) * filled_amount
        profit_pct = (fill_price - entry) / entry if side == "LONG" else (entry - fill_price) / entry
        fees = (filled_amount * entry * self.fee_rate) + (filled_amount * fill_price * self.fee_rate)
        exit_fee = filled_amount * fill_price * self.fee_rate
        net_pnl = pnl - fees
        realized_type = _realized_exit_type(exit_type or "TRAIL_WIN", net_pnl)

        if use_scale_out:
            remaining_amount = max(0.0, pos_amount - filled_amount)
            try:
                remaining_amount = float(self.exchange.amount_to_precision(self.symbol, remaining_amount))
            except Exception:
                remaining_amount = float(remaining_amount)

            if remaining_amount > 0 and remaining_amount < pos_amount:
                pos["amount"] = remaining_amount
                pos["runner_scale_out_taken"] = True
                pos["profit_runner_armed"] = False
                pos["trail_armed"] = True
                new_sl = _compute_trailing_stop(pos, current_price)
                if side == "LONG":
                    pos["sl"] = max(float(pos.get("sl", 0.0) or 0.0), new_sl)
                else:
                    pos["sl"] = min(float(pos.get("sl", 0.0) or 0.0), new_sl)
                pos["sl_pct_dist"] = abs(entry - float(pos.get("sl", entry) or entry)) / entry if entry else float(pos.get("sl_pct_dist", 0.0) or 0.0)
                self._log_trade(
                    trade_id,
                    "PARTIAL_EXIT",
                    order_side,
                    fill_price,
                    filled_amount,
                    pnl,
                    exit_fee,
                    t_type=realized_type,
                    reason="runner scale-out",
                )
                self._ensure_exchange_stop_loss(pos)
                self._ensure_exchange_take_profit(pos)
                self._ensure_native_trailing_stop(pos)
                self._last_trade_ts = time.time()
                self.last_status = (
                    f"{realized_type}: {side} scale-out {filled_amount:.4f} @ ${fill_price:.5f} "
                    f"Net ${net_pnl:+,.2f} | Rem {remaining_amount:.4f}"
                )
                logger.info(
                    f"[PARTIAL_EXIT] {side} scale-out filled: {filled_amount:.4f}/{pos_amount:.4f} @ {fill_price:.5f} "
                    f"| Net: {net_pnl:+.4f} ({profit_pct:+.2%}) | Remaining: {remaining_amount:.4f}"
                )
                return True

            use_scale_out = False

        self._record_closed_trade(realized_type, entry, fill_price, pnl, profit_pct * 100, fees)
        self._log_trade(trade_id, "EXIT", order_side, fill_price, filled_amount, pnl, exit_fee, t_type=realized_type)
        self.trade_count += 1
        self._last_trade_ts = time.time()
        self._recently_closed_ts = time.time()
        self._last_closed_side = side
        self._last_closed_trade_id = int(trade_id or 0)
        self._last_closed_trade_ts = time.time()
        if profit_pct > 0:
            self._last_profitable_exit_side = side
            self._last_profitable_exit_ts = time.time()
            self._opposite_reset_seen_after_profit = False
        self._cleanup_trade_orders(self.symbol, pos)
        self.last_status = f"{realized_type}: {side} @ ${fill_price:.5f} Net ${net_pnl:+,.2f}"
        logger.info(f"[MARKET_EXIT] {side} {filled_amount:.4f} @ {fill_price:.5f} | Net: {net_pnl:+.4f} ({profit_pct:+.2%})")
        return False

    def _cleanup_flat_protection_orders(self, symbol: str = None):
        """
        When flat, remove only reduce-only protection orders. Leave entry orders alone.
        """
        target_symbol = symbol or self.symbol
        cancelled = 0
        try:
            cancelled += self._cancel_reduce_only_orders("SELL", {"STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"})
            cancelled += self._cancel_reduce_only_orders("BUY", {"STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"})
        except Exception as e:
            logger.debug(f"Flat protection cleanup skipped: {e}")
        if cancelled:
            logger.info(f"[CLEANUP] Cancelled {cancelled} reduce-only protection order(s) while flat for {target_symbol}.")
        return cancelled

    def _cancel_non_reduce_open_orders(self, symbol: str = None):
        """
        Cancel stale entry orders while preserving reduce-only protection orders.
        """
        target_symbol = symbol or self.symbol
        try:
            for order in self.exchange.fetch_open_orders(target_symbol):
                info = order.get('info', {}) or {}
                reduce_only = _exchange_flag_true(order.get('reduceOnly')) or _exchange_flag_true(info.get('reduceOnly'))
                if reduce_only:
                    continue
                order_id = str(order.get('id') or info.get('orderId') or '')
                if order_id:
                    self.exchange.cancel_order(order_id, target_symbol)
        except Exception as e:
            logger.debug(f"Non-reduce order cleanup skipped: {e}")

    def emergency_close_all(self, symbol: str = None):
        """
        Emergency killswitch: Market close all positions and cancel all orders.
        """
        target_symbol = symbol or self.symbol
        logger.warning(f"EMERGENCY KILLSWITCH TRIGGERED for {target_symbol}")

        # 1. Cancel all pending orders
        self._cleanup_trade_orders(target_symbol)

        # 2. Market close any active position
        try:
            positions = self.exchange.fetch_positions([target_symbol])
            for pos in positions:
                if float(pos.get('contracts', 0) or 0) != 0:
                    side = 'SELL' if pos.get('side') == 'long' else 'BUY'
                    amount = abs(float(pos.get('contracts', 0)))
                    logger.info(f"Panic Closing {side} position: {amount} {target_symbol}")
                    self.exchange.create_order(
                        symbol=target_symbol,
                        type='MARKET',
                        side=side,
                        amount=amount,
                        params={'reduceOnly': True}
                    )
        except Exception as e:
            logger.error(f"Emergency position close failed: {e}")

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            target = symbol or self.symbol
            orders = self.exchange.fetch_open_orders(target)
            return [{
                'id': o.get('id'),
                'symbol': o.get('symbol'),
                'side': o.get('side', '').upper(),
                'price': o.get('price'),
                'amount': o.get('amount'),
                'filled': o.get('filled', 0.0),
                'type': o.get('type', '').upper()
            } for o in orders]
        except Exception as e:
            logger.debug(f"Open Orders Fetch Error: {e}")
            return []

    def get_portfolio_value(self, current_price: float) -> float:
        try:
            balance = self.exchange.fetch_balance()
            target_asset = _quote_asset_from_symbol(getattr(self, 'symbol', 'DOGE/USDT'))
            # On Futures, the real quote balance is often in the top-level currency bucket.
            total_equity = 0.0
            bucket = balance.get(target_asset)
            if isinstance(bucket, dict):
                for key in ('total', 'free', 'used'):
                    try:
                        total_equity = float(bucket.get(key, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        total_equity = 0.0
                    if total_equity > 0:
                        break

            if total_equity == 0:
                total_equity = float(balance.get('total', {}).get(target_asset, 0.0) or 0.0)

            if total_equity == 0 and 'info' in balance:
                info = balance['info']
                if 'assets' in info:
                    for asset in info['assets']:
                        if asset.get('asset') == target_asset:
                            for key in ('walletBalance', 'marginBalance', 'crossWalletBalance', 'availableBalance'):
                                try:
                                    total_equity = float(asset.get(key, 0.0) or 0.0)
                                except (TypeError, ValueError):
                                    total_equity = 0.0
                                if total_equity > 0:
                                    break
                            if total_equity > 0:
                                break
                if total_equity == 0:
                    for key in ('totalWalletBalance', 'totalMarginBalance', 'availableBalance', 'totalCrossWalletBalance'):
                        try:
                            total_equity = float(info.get(key, 0.0) or 0.0)
                        except (TypeError, ValueError):
                            total_equity = 0.0
                        if total_equity > 0:
                            break

            if total_equity == 0 and self._last_equity_value > 0:
                total_equity = float(self._last_equity_value)
            if total_equity == 0 and self.initial_balance > 0:
                unrealized = 0.0
                for pos in getattr(self, "active_positions", []) or []:
                    side = str(pos.get("side", "")).upper()
                    entry = float(pos.get("entry", 0.0) or 0.0)
                    amount = float(pos.get("amount", 0.0) or 0.0)
                    if side not in {"LONG", "SHORT"} or entry <= 0 or amount <= 0:
                        continue
                    unrealized += (current_price - entry) * amount if side == "LONG" else (entry - current_price) * amount
                total_equity = max(0.0, float(self.initial_balance) + unrealized)

            if not self._initial_price_set and total_equity > 0:
                self.initial_balance = total_equity
                self._initial_price_set = True
                logger.info(f"Initial Session Equity: ${self.initial_balance:,.2f}")
            if total_equity > 0:
                self._last_equity_value = total_equity
            return total_equity
        except Exception as e:
            logger.error(f"Equity Fetch Error: {e}")
            self.last_status = f"Equity error: {e}"
            return float(self._last_equity_value or self.initial_balance or 0.0)

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
        """Processes trailing stops for Binance Futures."""
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        self._last_price = current_price

        pending_entry = getattr(self, "pending_entry", None)
        if pending_entry and not self.active_positions:
            try:
                order_id = str(pending_entry.get("order_id") or "")
                age = time.time() - float(pending_entry.get("ts", time.time()) or time.time())
                ttl_seconds = float(pending_entry.get("ttl_seconds", getattr(self, "pending_entry_ttl_seconds", 20)) or 20)
                if age >= ttl_seconds:
                    if order_id:
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                        except Exception:
                            pass
                    self.pending_entry = None
                    self.last_status = "Pending entry expired"
            except Exception as e:
                logger.debug(f"Pending entry expiry skipped: {e}")

        pending_exit = getattr(self, "pending_exit", None)
        if pending_exit:
            try:
                order_id = str(pending_exit.get("order_id") or "")
                age = time.time() - float(pending_exit.get("ts", time.time()) or time.time())
                order = self.exchange.fetch_order(order_id, self.symbol) if order_id else {}
                status = str(order.get("status", "") or "").lower()
                filled = float(order.get("filled", 0.0) or 0.0)
                expected_amount = float(pending_exit.get("amount", 0.0) or 0.0)
                if status == "closed" or (expected_amount > 0 and filled >= expected_amount * 0.999):
                    fill_price = _order_fill_price(order, float(pending_exit.get("price", current_price) or current_price))
                    pos = self.active_positions[0] if self.active_positions else None
                    if pos:
                        entry = float(pos.get("entry", fill_price) or fill_price)
                        side = str(pos.get("side", "LONG"))
                        amount = float(pos.get("amount", expected_amount) or expected_amount)
                        pnl = (fill_price - entry) * amount if side == "LONG" else (entry - fill_price) * amount
                        profit_pct = (fill_price - entry) / entry if side == "LONG" else (entry - fill_price) / entry
                        fees = (amount * entry * self.fee_rate) + (amount * fill_price * self.fee_rate)
                        exit_fee = amount * fill_price * self.fee_rate
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type(str(pending_exit.get("exit_type", "TRAIL_WIN") or "TRAIL_WIN"), net_pnl)
                        self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
                        self._log_trade(pos.get("trade_id", 0), "EXIT", pending_exit.get("side", "SELL"), fill_price, amount, pnl, exit_fee, t_type=exit_type)
                        self.trade_count += 1
                        self._last_trade_ts = time.time()
                        self._recently_closed_ts = time.time()
                        self._last_closed_side = side
                        self._cleanup_trade_orders(self.symbol, pos)
                        self._post_close_cleanup_needed = True
                        self.active_positions = []
                        self.pending_exit = None
                        self._last_closed_trade_id = int(pos.get("trade_id", 0) or 0)
                        self._last_closed_trade_ts = time.time()
                        self.last_status = f"{exit_type}: {side} @ ${fill_price:.5f} Net ${net_pnl:+,.2f}"
                        logger.info(f"[ORDER] Maker exit FILLED: {side} {amount:.4f} @ {fill_price:.5f} | Net: {net_pnl:+.4f} ({profit_pct:+.2%})")
                        return
                else:
                    ttl_seconds = float(pending_exit.get("ttl_seconds", getattr(self, "pending_exit_ttl_seconds", 20)) or 20)
                    if age < ttl_seconds:
                        self.last_status = f"Waiting maker exit fill @ ${float(pending_exit.get('price', current_price) or current_price):.5f}"
                        return
                    if order_id:
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                        except Exception:
                            pass
                    fallback_market = _exchange_flag_true(
                        pending_exit.get("fallback_market", getattr(self, "market_fallback_on_timeout", False))
                    )
                    if not fallback_market:
                        pos = self.active_positions[0] if self.active_positions else None
                        if pos:
                            self._ensure_exchange_stop_loss(pos)
                            self._ensure_exchange_take_profit(pos)
                            self._ensure_native_trailing_stop(pos)
                        self.pending_exit = None
                        self.last_status = "Maker exit expired; reprice"
                        return
                    pos = self.active_positions[0] if self.active_positions else None
                    refreshed_amount = None
                    if pos:
                        try:
                            positions = self.exchange.fetch_positions()
                            for p in positions:
                                if p["symbol"] == self.symbol and float(p.get("contracts", 0)) != 0:
                                    exch_amount = abs(float(p.get("contracts", 0)))
                                    if exch_amount > 0:
                                        refreshed_amount = exch_amount
                                        pos = dict(pos)
                                        pos["amount"] = exch_amount
                                        pos["side"] = "LONG" if float(p.get("contracts", 0)) > 0 else "SHORT"
                                        try:
                                            pos["entry"] = float(p.get("entryPrice", pos.get("entry", current_price)) or pos.get("entry", current_price))
                                        except Exception:
                                            pass
                                    break
                        except Exception as refresh_err:
                            logger.debug(f"Exit fallback position refresh skipped: {refresh_err}")
                    if not pos:
                        self.pending_exit = None
                        self.last_status = "Maker exit expired; no open position"
                        return
                    filled = float(order.get("filled", 0.0) or 0.0)
                    expected_amount = float(pending_exit.get("amount", 0.0) or 0.0)
                    partial_amount = min(expected_amount, filled) if filled > 0 else 0.0
                    if refreshed_amount is None and partial_amount > 0:
                        pos = dict(pos)
                        pos["amount"] = max(0.0, expected_amount - partial_amount)
                    if partial_amount > 0:
                        side = str(pos.get("side", "LONG"))
                        entry = float(pos.get("entry", current_price) or current_price)
                        fill_price = _order_fill_price(order, float(pending_exit.get("price", current_price) or current_price))
                        pnl = (fill_price - entry) * partial_amount if side == "LONG" else (entry - fill_price) * partial_amount
                        profit_pct = (fill_price - entry) / entry if side == "LONG" else (entry - fill_price) / entry
                        fees = (partial_amount * entry * self.fee_rate) + (partial_amount * fill_price * self.fee_rate)
                        exit_fee = partial_amount * fill_price * self.fee_rate
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type(str(pending_exit.get("exit_type", "TRAIL_WIN") or "TRAIL_WIN"), net_pnl)
                        self._log_trade(
                            pos.get("trade_id", 0),
                            "PARTIAL_EXIT",
                            pending_exit.get("side", "SELL"),
                            fill_price,
                            partial_amount,
                            pnl,
                            exit_fee,
                            t_type=exit_type,
                            reason="maker exit partial fill before market fallback",
                        )
                        self._last_trade_ts = time.time()
                        self.last_status = (
                            f"{exit_type}: {side} partial maker fill {partial_amount:.4f} @ ${fill_price:.5f} "
                            f"Net ${net_pnl:+,.2f}; falling back"
                        )
                    self.pending_exit = None
                    self._last_closed_trade_id = int(pos.get("trade_id", 0) or 0)
                    self._last_closed_trade_ts = time.time()
                    return self._finalize_reduce_only_market_exit(
                        pos,
                        current_price,
                        str(pending_exit.get("exit_type", "TRAIL_WIN") or "TRAIL_WIN"),
                        int(pos.get("trade_id", 0) or 0),
                        amount=None,
                        use_scale_out=False,
                    )
            except Exception as e:
                logger.debug(f"Pending exit check skipped: {e}")

        # Sync positions with exchange
        try:
            positions = self.exchange.fetch_positions()
            self._last_position_sync_ok = True
            exch_pos = None
            for p in positions:
                # Binance Futures symbol matching
                if p['symbol'] == self.symbol and float(p.get('contracts', 0)) != 0:
                    exch_pos = p
                    break

            if exch_pos is None:
                # No position on exchange. Preserve live maker entries while they
                # are still within TTL, but remove stale reduce-only protection
                # orders left behind by a completed/externally closed position.
                try:
                    open_orders = self.exchange.fetch_open_orders(self.symbol) or []
                    protection_orders = self._fetch_open_protection_orders(self.symbol)
                    order_count = max(len(open_orders), len(protection_orders))
                    if order_count:
                        if self.active_positions or getattr(self, "pending_exit", None):
                            logger.info(f"[SYNC] Exchange flat with {order_count} open order(s); purging closed-position orders for {self.symbol}.")
                            self._cleanup_trade_orders(self.symbol, self.active_positions[0] if self.active_positions else None)
                            self._wipe_all_orphans(self.symbol)
                        elif getattr(self, "pending_entry", None):
                            logger.info(f"[SYNC] Exchange flat with pending entry; preserving entry and clearing stale protection for {self.symbol}.")
                            self._cleanup_flat_protection_orders(self.symbol)
                        else:
                            logger.info(f"[SYNC] Exchange flat with {order_count} open orphan order(s); clearing stale orders for {self.symbol}.")
                            self._cleanup_trade_orders(self.symbol)
                            self._wipe_all_orphans(self.symbol)
                except Exception as e:
                    logger.debug(f"[SYNC] Pre-clean flat orphan purge skipped: {e}")

                if self.active_positions or getattr(self, "pending_exit", None):
                    logger.info(f"[SYNC] No position on exchange for {self.symbol}, clearing local state.")
                    if self.active_positions:
                        pos = self.active_positions[0]
                        trade_id = int(pos.get("trade_id", 0) or 0)
                        recent_closed_id = int(getattr(self, "_last_closed_trade_id", 0) or 0)
                        recent_closed_ts = float(getattr(self, "_last_closed_trade_ts", 0.0) or 0.0)
                        if trade_id and trade_id == recent_closed_id and (time.time() - recent_closed_ts) < 30:
                            logger.info(
                                f"[SYNC] Suppressing duplicate close record for trade_id={trade_id}; already finalized locally."
                            )
                            self.active_positions = []
                            self.pending_exit = None
                            self._post_close_cleanup_needed = True
                            if getattr(self, "pending_entry", None):
                                self.last_status = "Flat; preserving pending entry"
                            return
                        fill_price = self._resolve_exchange_close_fill_price(pos, current_price)
                        self._log_exchange_close_diagnostics(pos, current_price)
                        entry = float(pos.get("entry", fill_price) or fill_price)
                        amount = float(pos.get("amount", 0.0) or 0.0)
                        side = str(pos.get("side", "LONG") or "LONG").upper()
                        pnl = (fill_price - entry) * amount if side == "LONG" else (entry - fill_price) * amount
                        profit_pct = (fill_price - entry) / entry if side == "LONG" else (entry - fill_price) / entry
                        fees = (amount * entry * self.fee_rate) + (amount * fill_price * self.fee_rate)
                        exit_fee = amount * fill_price * self.fee_rate
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type("EXCHANGE_CLOSED", net_pnl)
                        order_side = "SELL" if side == "LONG" else "BUY"
                        self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
                        self._log_trade(trade_id, "EXIT", order_side, fill_price, amount, pnl, exit_fee, t_type=exit_type)
                        self.trade_count += 1
                        self._last_trade_ts = time.time()
                        self._recently_closed_ts = time.time()
                        self._last_closed_side = side
                        self._last_closed_trade_id = trade_id
                        self._last_closed_trade_ts = time.time()
                    try:
                        self._cleanup_trade_orders(self.symbol, self.active_positions[0] if self.active_positions else None)
                    except Exception as e:
                        logger.debug(f"[SYNC] Open-order cleanup skipped: {e}")
                    self._post_close_cleanup_needed = True

                    # MUST clear local state immediately to prevent infinite loop of recording closes
                    self.active_positions = []
                    self.pending_exit = None
                if self._post_close_cleanup_needed:
                    try:
                        self._cleanup_flat_protection_orders(self.symbol)
                    except Exception as e:
                        logger.debug(f"[SYNC] Flat protection cleanup skipped: {e}")
                    finally:
                        self._post_close_cleanup_needed = False

                # RUTHLESS FLAT CLEANUP: If exchange says we are flat, we must be flat.
                # Keep pending entry orders alive; do not wipe them just because we're flat.
                if getattr(self, "pending_entry", None):
                    self.last_status = "Flat; preserving pending entry"
            else:
                # Position exists on exchange
                exch_size = abs(float(exch_pos.get('contracts', 0)))
                exch_side = 'LONG' if float(exch_pos.get('contracts', 0)) > 0 else 'SHORT'
                # CCXT side can also be 'long'/'short'
                if exch_pos.get('side'):
                    exch_side = exch_pos['side'].upper()

                entry_price = float(exch_pos.get('entryPrice', current_price))

                # GHOST SHIELD: Don't adopt a position if we just closed one with the same side/size
                # This prevents double-counting due to Binance API lag.
                recently_closed = getattr(self, '_recently_closed_ts', 0)
                if not self.active_positions and (time.time() - recently_closed) < 30:
                    last_side = getattr(self, '_last_closed_side', '')
                    if exch_side == last_side:
                        logger.debug(f"[SYNC] Ignoring ghost position for {self.symbol} (Recently closed)")
                        return

                if not self.active_positions:
                    # ADOPT position: It's on exchange but not in our local memory (e.g. after restart)
                    logger.info(f"[SYNC] Adopting existing {exch_side} position for {self.symbol} (Size: {exch_size}, Entry: {entry_price})")
                    pending_entry = getattr(self, "pending_entry", None)
                    pending_sl = None
                    pending_tp = None
                    pending_support = None
                    pending_resistance = None
                    pending_pivot_classic = None
                    pending_trade_id = 0
                    pending_score = 0.0
                    pending_confidence = 0.0
                    pending_reason = ""
                    if isinstance(pending_entry, dict):
                        pending_action = str(pending_entry.get("action", "") or "").upper()
                        pending_side = "LONG" if pending_action == "BUY" else ("SHORT" if pending_action == "SELL" else "")
                        if pending_side == exch_side:
                            pending_sl = pending_entry.get("sl")
                            pending_tp = pending_entry.get("tp")
                            pending_support = pending_entry.get("structure_support")
                            pending_resistance = pending_entry.get("structure_resistance")
                            pending_pivot_classic = pending_entry.get("pivot_classic")
                            pending_trade_id = int(pending_entry.get("trade_id", 0) or 0)
                            pending_score = float(pending_entry.get("score", 0.0) or 0.0)
                            pending_confidence = float(pending_entry.get("confidence", 0.0) or 0.0)
                            pending_reason = str(pending_entry.get("reason", "") or "")
                            self.pending_entry = None
                    initial_sl = _safe_initial_sl_price(
                        exch_side,
                        entry_price,
                        pending_sl,
                        getattr(self, 'default_sl_pct', 0.0030),
                        pending_pivot_classic,
                        max_sl_pct=getattr(self, 'max_structural_sl_pct', 0.0120),
                    )
                    tp_price = _safe_tp_price(
                        exch_side,
                        entry_price,
                        pending_tp,
                        float(getattr(self, "tp_pct", 0.0025)),
                    )
                    runner_enabled = bool((getattr(self, "scalp_config", {}) or {}).get("runner_enabled", True))
                    tp_price = _runner_emergency_tp_price(
                        exch_side,
                        entry_price,
                        tp_price,
                        getattr(self, "scalp_config", {}) or {},
                    )

                    adopted_trade_id = pending_trade_id
                    if adopted_trade_id <= 0:
                        adopted_trade_id = int(getattr(self, "_next_trade_id", 1) or 1)
                        self._next_trade_id = adopted_trade_id + 1
                    else:
                        self._next_trade_id = max(int(getattr(self, "_next_trade_id", 1) or 1), adopted_trade_id + 1)

                    adopted_pos = {
                        'trade_id': adopted_trade_id,
                        'side': exch_side,
                        'entry': entry_price,
                        'amount': exch_size,
                        'entry_ts': time.time(),
                        'highest_price': current_price if exch_side == 'LONG' else 0,
                        'lowest_price': current_price if exch_side == 'SHORT' else 0,
                        'highest_profit_pct': 0.0,
                        'sl': initial_sl,
                        'tp_price': tp_price,
                        'fixed_take_profit_enabled': not runner_enabled,
                        'sl_pct_dist': abs(entry_price - initial_sl) / entry_price if entry_price else 0.0050,
                        'fee_rate': getattr(self, 'fee_rate', 0.0004),
                        'min_profit_after_fees': getattr(self, 'min_profit_after_fees', 0.0005),
                        'break_even_trigger_pct': float(getattr(self, 'break_even_trigger_pct', 0.0010)),
                        'break_even_buffer_pct': float(getattr(self, 'break_even_buffer_pct', 0.0002)),
                        'profit_trailing_enabled': bool(getattr(self, 'profit_trailing_enabled', True)),
                        'profit_trailing_activation_pct': float(getattr(self, 'profit_trailing_activation_pct', getattr(self, 'break_even_trigger_pct', 0.0010))),
                        'trailing_tp_enabled': bool(getattr(self, 'trailing_tp_enabled', True)),
                        'trailing_tp_giveback_pct': float(getattr(self, 'trailing_tp_giveback_pct', 0.12)),
                        'trailing_tp_min_peak_pct': float(getattr(self, 'trailing_tp_min_peak_pct', getattr(self, 'profit_trailing_activation_pct', 0.0020))),
                        'trail_tighten_1_pct': float(getattr(self, 'trail_tighten_1_pct', 0.0030)),
                        'trail_tighten_2_pct': float(getattr(self, 'trail_tighten_2_pct', 0.0060)),
                        'trail_t1_gap_pct': float(getattr(self, 'trail_t1_gap_pct', 0.0025)),
                        'trail_t2_gap_pct': float(getattr(self, 'trail_t2_gap_pct', 0.0020)),
                        'trail_armed': False,
                        'structure_support': pending_support,
                        'structure_resistance': pending_resistance,
                        'native_trailing_activation_pct': float(getattr(self, 'profit_trailing_activation_pct', getattr(self, 'break_even_trigger_pct', 0.0010))),
                    }
                    logger.warning(f"[ADOPT] Found existing {exch_side} position. Adopting with safety SL at {initial_sl:.4f}")
                    self.active_positions.append(adopted_pos)
                    entry_side = "BUY" if exch_side == "LONG" else "SELL"
                    entry_fee = exch_size * entry_price * self.fee_rate
                    self._log_trade(
                        adopted_trade_id,
                        "ENTRY",
                        entry_side,
                        entry_price,
                        exch_size,
                        fees=entry_fee,
                        score=pending_score,
                        confidence=pending_confidence,
                        reason=pending_reason or "adopted existing exchange position",
                        t_type="ADOPTED",
                    )
                else:
                    # SYNC position: Already tracking, just ensure size/side matches
                    local_pos = self.active_positions[0]
                    if abs(exch_size - local_pos['amount']) > 0.000001 or exch_side != local_pos['side']:
                        logger.warning(f"[SYNC] Position mismatch: local {local_pos['side']} {local_pos['amount']}, exchange {exch_side} {exch_size}")
                        local_pos['amount'] = exch_size
                        local_pos['side'] = exch_side
        except Exception as e:
            self._last_position_sync_ok = False
            logger.warning(f"Failed to sync positions: {e}")

        if len(self.active_positions) > 1:
            primary = next((p for p in self.active_positions if int(p.get('trade_id', 0) or 0) > 0), self.active_positions[0])
            for extra in self.active_positions[1:]:
                if extra is primary:
                    continue
                trail_id = str(extra.get('native_trailing_order_id') or '')
                if trail_id:
                    try:
                        self.exchange.cancel_order(trail_id, self.symbol)
                    except Exception:
                        pass
            self.active_positions = [primary]
            self.last_status = "Collapsed duplicate local position tracking"

        if self.active_positions and not getattr(self, 'dca_enabled', False):
            self._cancel_non_reduce_open_orders(self.symbol)

        if getattr(self, "_last_position_sync_ok", False):
            self._last_sync_ts = time.time()

        remaining = []
        try:
            for pos in self.active_positions:
                closed = False
                entry = pos['entry']
                side = pos['side']
                amount = pos['amount']
                trade_id = pos.get("trade_id", 0)

                self._reset_excess_protection_orders(pos)
                self._ensure_exchange_stop_loss(pos)
                self._ensure_exchange_take_profit(pos)
                if getattr(self, 'use_native_trailing_stop', False):
                    self._ensure_native_trailing_stop(pos)

                exit_type = ""
                # Trailing stop now follows best price reached and moves to breakeven early.
                if side == 'LONG':
                    if current_price > float(pos.get('highest_price', entry) or entry):
                        pos['highest_price'] = current_price

                    profit_pct = (current_price - entry) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', getattr(self, 'min_profit_after_fees', 0.0002)))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit
                    tp_price = float(pos.get('tp_price', 0.0) or 0.0)
                    trail_activation = float(pos.get("profit_trailing_activation_pct", pos.get("break_even_trigger_pct", 0.0020)) or 0.0020)
                    if bool(pos.get("profit_trailing_enabled", True)):
                        trail_activation = max(trail_activation, float(pos.get("break_even_trigger_pct", 0.0020) or 0.0020))
                    if not bool(pos.get("trail_armed", False)) and profit_pct >= trail_activation:
                        pos["trail_armed"] = True
                        self._ensure_native_trailing_stop(pos)

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    if not closed and bool(pos.get("fixed_take_profit_enabled", True)) and tp_price > 0 and current_price >= tp_price and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TAKE_PROFIT"

                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            min_hold = int(scalp_cfg.get('min_hold_seconds', 10))
                            highest_profit = float(pos.get('highest_profit_pct', 0.0) or 0.0)
                            runner_pullback = float(scalp_cfg.get('runner_pullback_pct', 0.0012) or 0.0012)
                            runner_lock = max(
                                min_net_profit,
                                float(scalp_cfg.get('runner_min_lock_pct', 0.0018) or 0.0018),
                            )
                            runner_enabled = bool(scalp_cfg.get("runner_enabled", True))
                            if runner_enabled and profit_pct >= tp_pct and hold_time >= min_hold and profit_pct >= min_net_profit:
                                was_armed = bool(pos.get("profit_runner_armed", False))
                                pos["profit_runner_armed"] = True
                                protected_sl = entry * (1 + runner_lock)
                                if protected_sl > float(pos['sl']):
                                    pos['sl'] = min(protected_sl, current_price * 0.9995)
                                    self._ensure_exchange_stop_loss(pos)
                                if not was_armed:
                                    logger.info(
                                        f"[PROFIT_RUNNER] LONG armed at {profit_pct:+.2%}; protected stop ${float(pos['sl']):.5f}"
                                    )

                            macd_diff_now = float(getattr(self, "_current_macd_diff", 0.0) or 0.0)
                            psar_now = getattr(self, "_current_psar", None)
                            runner_pullback_hit = bool(pos.get("profit_runner_armed")) and highest_profit > profit_pct and (highest_profit - profit_pct) >= runner_pullback
                            runner_reversal_hit = bool(pos.get("profit_runner_armed")) and (
                                macd_diff_now < 0 or (psar_now is not None and float(psar_now) > current_price)
                            )
                            if runner_enabled and bool(pos.get("profit_runner_armed")) and profit_pct >= runner_lock and (runner_pullback_hit or runner_reversal_hit):
                                if bool(pos.get("trailing_tp_enabled", True)):
                                    pos["trail_armed"] = True
                                    if not bool(pos.get("scalp_signal_trail_seen", False)):
                                        pos["scalp_signal_trail_seen"] = True
                                        logger.info("[PROFIT_RUNNER] LONG scalp signal tightened protection; trailing TP remains primary.")
                                else:
                                    closed = True
                                    exit_type = "SCALP_EXIT"
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_FADE"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    ttl_armed = bool(pos.get("profit_runner_armed", False))
                    ttl_allows_runner = profit_pct < float(pos.get("break_even_trigger_pct", 0.0020) or 0.0020)
                    if (
                        not closed
                        and ttl_seconds > 0
                        and hold_time > ttl_seconds
                        and profit_pct >= min_net_profit
                        and (not ttl_armed)
                        and ttl_allows_runner
                    ):
                        closed = True
                        exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        if new_sl > float(pos['sl']):
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Support Lock)")
                            self._ensure_exchange_stop_loss(pos)

                    if not closed and current_price <= pos['sl']:
                        closed = True
                        exit_type = "STOP_LOSS"

                else: # SHORT
                    if current_price < float(pos.get('lowest_price', entry) or entry):
                        pos['lowest_price'] = current_price

                    profit_pct = (entry - current_price) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', getattr(self, 'min_profit_after_fees', 0.0002)))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit
                    tp_price = float(pos.get('tp_price', 0.0) or 0.0)
                    trail_activation = float(pos.get("profit_trailing_activation_pct", pos.get("break_even_trigger_pct", 0.0020)) or 0.0020)
                    if bool(pos.get("profit_trailing_enabled", True)):
                        trail_activation = max(trail_activation, float(pos.get("break_even_trigger_pct", 0.0020) or 0.0020))
                    if not bool(pos.get("trail_armed", False)) and profit_pct >= trail_activation:
                        pos["trail_armed"] = True
                        self._ensure_native_trailing_stop(pos)

                    if not closed and _trailing_tp_hit(pos, profit_pct, min_net_profit):
                        closed = True
                        exit_type = "TRAIL_TP"

                    if not closed and bool(pos.get("fixed_take_profit_enabled", True)) and tp_price > 0 and current_price <= tp_price and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TAKE_PROFIT"

                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            min_hold = int(scalp_cfg.get('min_hold_seconds', 10))
                            highest_profit = float(pos.get('highest_profit_pct', 0.0) or 0.0)
                            runner_pullback = float(scalp_cfg.get('runner_pullback_pct', 0.0012) or 0.0012)
                            runner_lock = max(
                                min_net_profit,
                                float(scalp_cfg.get('runner_min_lock_pct', 0.0018) or 0.0018),
                            )
                            runner_enabled = bool(scalp_cfg.get("runner_enabled", True))
                            if runner_enabled and profit_pct >= tp_pct and hold_time >= min_hold and profit_pct >= min_net_profit:
                                was_armed = bool(pos.get("profit_runner_armed", False))
                                pos["profit_runner_armed"] = True
                                protected_sl = entry * (1 - runner_lock)
                                if protected_sl < float(pos['sl']):
                                    pos['sl'] = max(protected_sl, current_price * 1.0005)
                                    self._ensure_exchange_stop_loss(pos)
                                if not was_armed:
                                    logger.info(
                                        f"[PROFIT_RUNNER] SHORT armed at {profit_pct:+.2%}; protected stop ${float(pos['sl']):.5f}"
                                    )

                            macd_diff_now = float(getattr(self, "_current_macd_diff", 0.0) or 0.0)
                            psar_now = getattr(self, "_current_psar", None)
                            runner_pullback_hit = bool(pos.get("profit_runner_armed")) and highest_profit > profit_pct and (highest_profit - profit_pct) >= runner_pullback
                            runner_reversal_hit = bool(pos.get("profit_runner_armed")) and (
                                macd_diff_now > 0 or (psar_now is not None and float(psar_now) < current_price)
                            )
                            if runner_enabled and bool(pos.get("profit_runner_armed")) and profit_pct >= runner_lock and (runner_pullback_hit or runner_reversal_hit):
                                if bool(pos.get("trailing_tp_enabled", True)):
                                    pos["trail_armed"] = True
                                    if not bool(pos.get("scalp_signal_trail_seen", False)):
                                        pos["scalp_signal_trail_seen"] = True
                                        logger.info("[PROFIT_RUNNER] SHORT scalp signal tightened protection; trailing TP remains primary.")
                                else:
                                    closed = True
                                    exit_type = "SCALP_EXIT"
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_FADE"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    ttl_armed = bool(pos.get("profit_runner_armed", False))
                    ttl_allows_runner = profit_pct < float(pos.get("break_even_trigger_pct", 0.0020) or 0.0020)
                    if (
                        not closed
                        and ttl_seconds > 0
                        and hold_time > ttl_seconds
                        and profit_pct >= min_net_profit
                        and (not ttl_armed)
                        and ttl_allows_runner
                    ):
                        closed = True
                        exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        if new_sl < float(pos['sl']):
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Resistance Lock)")
                            self._ensure_exchange_stop_loss(pos)

                    if not closed and current_price >= pos['sl']:
                        closed = True
                        exit_type = "STOP_LOSS"

                if closed:
                    logger.info(f"[FUTURES] Exit {side} @ {current_price:.2f}")
                    keep_position = self._execute_futures_reduce_only_exit(pos, current_price, exit_type, trade_id)
                    if keep_position:
                        remaining.append(pos)
                else:
                    remaining.append(pos)
            self.active_positions = remaining
        except Exception as e:
            logger.error(f"Futures Process Error: {e}")
            self.last_status = f"Process error: {e}"

    def place_limit_order(self, signal: dict, symbol: str, current_price: float):
        """Handles Long/Short Entries and Reversals on Binance Futures."""
        if getattr(self, "paused", False):
            self.last_status = "Trading PAUSED"
            return
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        action = signal['action']
        self.observe_signal_cycle(signal)
        if action == "HOLD":
            self.last_status = "Signal HOLD"
            return

        try:
            now = time.time()
            if not getattr(self, '_last_position_sync_ok', False):
                self.last_status = "Veto: position sync not confirmed"
                return
            if getattr(self, "pending_exit", None):
                self.last_status = "Waiting pending exit"
                return
            pending_entry = getattr(self, "pending_entry", None)
            # Guard: cancel stale pending entry if older than TTL
            if pending_entry:
                age = now - float(pending_entry.get("ts", now) or now)
                ttl_seconds = float(pending_entry.get("ttl_seconds", getattr(self, "pending_entry_ttl_seconds", 20)) or 20)
                if age > ttl_seconds:
                    order_id = str(pending_entry.get("order_id", ""))
                    if order_id:
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                            logger.info(f"[ORDER] Cancelled stale pending entry #{order_id[-8:]} (age: {int(age)}s)")
                        except Exception:
                            pass
                    self.pending_entry = None
                    pending_entry = None
                    self.last_status = "Cleaned stale entry order"
            if pending_entry:
                try:
                    order_id = str(pending_entry.get("order_id") or "")
                    if self.active_positions and not getattr(self, 'dca_enabled', False):
                        if order_id:
                            try:
                                self.exchange.cancel_order(order_id, self.symbol)
                            except Exception:
                                pass
                        self.pending_entry = None
                        self.last_status = "Cancelled duplicate entry while in trade"
                        return
                    age = now - float(pending_entry.get("ts", now) or now)
                    order = self.exchange.fetch_order(order_id, self.symbol) if order_id else {}
                    status = str(order.get("status", "") or "").lower()
                    filled = float(order.get("filled", 0.0) or 0.0)
                    expected_amount = float(pending_entry.get("amount", 0.0) or 0.0)
                    if status == "closed" or (expected_amount > 0 and filled >= expected_amount * 0.999):
                        fill_price = _order_fill_price(order, float(pending_entry.get("price", current_price) or current_price))
                        action = str(pending_entry.get("action", action) or action)
                        trade_id = int(pending_entry.get("trade_id", 0) or 0)
                        amount = float(pending_entry.get("amount", 0.0) or 0.0)
                        pos_side = 'LONG' if action == "BUY" else 'SHORT'
                        # Try smart SL first (structure + ATR buffer), fallback to default
                        support = float(pending_entry.get("structure_support", 0.0) or 0.0)
                        resistance = float(pending_entry.get("structure_resistance", 0.0) or 0.0)
                        atr = float(pending_entry.get("atr", getattr(self, "_current_atr_pct", 0.0) * fill_price) or 0.0)
                        smart_sl = _smart_sl_price(pos_side, fill_price, support, resistance, atr, atr_multiplier=1.5)
                        default_sl = smart_sl
                        logger.info(f"[SMART_SL] {pos_side} entry {fill_price:.2f}: support={support:.2f} resistance={resistance:.2f} ATR={atr:.4f} → SL {smart_sl:.2f}")
                        sl_price = _safe_initial_sl_price(
                            pos_side,
                            fill_price,
                            pending_entry.get("sl", default_sl),
                            getattr(self, 'default_sl_pct', 0.0030),
                            pending_entry.get("pivot_classic"),
                            max_sl_pct=getattr(self, 'max_structural_sl_pct', 0.0120),
                        )
                        try:
                            tp_price = float(pending_entry.get("tp")) if pending_entry.get("tp") else None
                        except (TypeError, ValueError):
                            tp_price = None
                        if tp_price:
                            tp_price = _runner_emergency_tp_price(
                                pos_side,
                                fill_price,
                                tp_price,
                                getattr(self, "scalp_config", {}) or {},
                            )
                        runner_enabled = bool((getattr(self, "scalp_config", {}) or {}).get("runner_enabled", True))
                        if self.active_positions:
                            existing = self.active_positions[0]
                            if str(existing.get('side', '')).upper() == pos_side:
                                existing['trade_id'] = existing.get('trade_id') or trade_id
                                existing['entry'] = fill_price
                                existing['amount'] = amount
                                existing['trade_usdt'] = float(pending_entry.get('trade_usdt', existing.get('trade_usdt', 0.0)) or 0.0)
                                existing['effective_leverage'] = float(pending_entry.get('effective_leverage', existing.get('effective_leverage', getattr(self, 'leverage', 1.0))) or getattr(self, 'leverage', 1.0))
                                existing['sl'] = sl_price
                                existing['sl_pct_dist'] = abs(fill_price - sl_price) / fill_price if fill_price else existing.get('sl_pct_dist', 0.005)
                                existing['tp_price'] = tp_price
                                existing['fixed_take_profit_enabled'] = not runner_enabled
                                existing['structure_support'] = pending_entry.get('structure_support')
                                existing['structure_resistance'] = pending_entry.get('structure_resistance')
                                existing['profit_trailing_enabled'] = bool(getattr(self, 'profit_trailing_enabled', True))
                                existing['profit_trailing_activation_pct'] = float(getattr(self, 'profit_trailing_activation_pct', self.break_even_trigger_pct))
                                existing['trailing_tp_enabled'] = bool(getattr(self, 'trailing_tp_enabled', True))
                                existing['trailing_tp_giveback_pct'] = float(getattr(self, 'trailing_tp_giveback_pct', 0.12))
                                existing['trailing_tp_min_peak_pct'] = float(getattr(self, 'trailing_tp_min_peak_pct', getattr(self, 'profit_trailing_activation_pct', 0.0020)))
                                existing['native_trailing_activation_pct'] = float(getattr(self, 'profit_trailing_activation_pct', self.break_even_trigger_pct))
                                existing['trail_armed'] = bool(existing.get('trail_armed', False))
                                self._ensure_exchange_stop_loss(existing)
                                self._ensure_exchange_take_profit(existing)
                                self._ensure_native_trailing_stop(existing)
                                if not getattr(self, 'dca_enabled', False):
                                    self._cancel_non_reduce_open_orders(self.symbol)
                                self.pending_entry = None
                                self.last_status = f"Maker entry synced @ {fill_price:.5f}"
                                return
                        filled_pos = {
                            'trade_id': trade_id,
                            'side': pos_side,
                            'entry': fill_price,
                            'amount': amount,
                            'trade_usdt': float(pending_entry.get('trade_usdt', 0.0) or 0.0),
                            'effective_leverage': float(pending_entry.get('effective_leverage', getattr(self, 'leverage', 1.0)) or getattr(self, 'leverage', 1.0)),
                            'entry_ts': now,
                            'hold_until_ts': float(pending_entry.get("hold_until_ts", 0.0) or 0.0),
                            'highest_price': fill_price if action == "BUY" else 0,
                            'lowest_price': fill_price if action == "SELL" else 0,
                            'highest_profit_pct': 0.0,
                            'sl_pct_dist': abs(fill_price - sl_price) / fill_price if fill_price else 0.005,
                            'fee_rate': float(self.fee_rate),
                            'min_profit_after_fees': float(getattr(self, 'min_profit_after_fees', 0.0001)),
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
                            'fixed_take_profit_enabled': not runner_enabled,
                            'structure_support': pending_entry.get('structure_support'),
                            'structure_resistance': pending_entry.get('structure_resistance'),
                            'native_trailing_activation_pct': float(getattr(self, 'profit_trailing_activation_pct', getattr(self, 'break_even_trigger_pct', 0.0020))),
                        }
                        self.active_positions.append(filled_pos)
                        self._ensure_exchange_stop_loss(filled_pos)
                        self._ensure_exchange_take_profit(filled_pos)
                        self._ensure_native_trailing_stop(filled_pos)
                        if not getattr(self, 'dca_enabled', False):
                            self._cancel_non_reduce_open_orders(self.symbol)
                        self.pending_entry = None
                        self.last_status = f"Maker entry filled @ {fill_price:.5f}"
                        logger.info(f"[ORDER] Maker entry FILLED: {action} {amount:.4f} @ {fill_price:.5f}")
                        return
                    elif age < float(pending_entry.get("ttl_seconds", getattr(self, "pending_entry_ttl_seconds", 3)) or 3):
                        self.last_status = f"Waiting entry fill @ ${float(pending_entry.get('price', current_price) or current_price):.5f}"
                        return
                    else:
                        # Maker entry failed to fill in time. Cancel and wait for a fresh signal.
                        if order_id:
                            try:
                                self.exchange.cancel_order(order_id, self.symbol)
                            except Exception:
                                pass
                        if getattr(self, "market_fallback_on_timeout", False):
                            logger.warning("Market fallback is enabled but disabled in live-first safety path.")
                        self.pending_entry = None
                        self.last_status = "Maker entry expired; no market fallback"
                        return
                except Exception as e:
                    logger.debug(f"Pending entry check skipped: {e}")

            if self.min_seconds_between_trades and (now - float(self._last_trade_ts)) < float(self.min_seconds_between_trades):
                self.last_status = f"Cooldown ({int(self.min_seconds_between_trades)}s)"
                return
            if now < float(getattr(self, "_entry_block_until_ts", 0.0) or 0.0):
                wait_s = int(max(1.0, float(getattr(self, "_entry_block_until_ts", 0.0) - now)))
                self.last_status = f"Entry backoff ({wait_s}s)"
                return

            # Fee-aware minimum edge filter (TP distance must clear estimated costs)
            try:
                tp = float(signal.get("tp", 0.0) or 0.0)
                expected_tp_pct = abs(tp - float(current_price)) / float(current_price) if tp > 0 and current_price else float(getattr(self, 'tp_pct', 0.0030))
                roundtrip_cost_pct = (2.0 * float(self.fee_rate)) + float(self.fee_slippage_buffer_pct)
                if roundtrip_cost_pct > 0 and expected_tp_pct < (float(self.fee_edge_multiplier) * roundtrip_cost_pct):
                    self.last_status = "Veto: edge < fees"
                    return
            except Exception:
                pass

            # 1. Reversal Handling
            if self.active_positions:
                current_pos = self.active_positions[0]
                if (action == "SELL" and current_pos['side'] == "LONG") or (action == "BUY" and current_pos['side'] == "SHORT"):
                    hold_until = float(current_pos.get("hold_until_ts", 0.0) or 0.0)
                    if hold_until and now < hold_until:
                        self.last_status = "Veto: hold period"
                        return
                    entry_ts = float(current_pos.get("entry_ts", now))
                    age = now - entry_ts
                    # REVERSAL HANDLING: require enough edge to pay both exit and re-entry costs.
                    profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']
                    round_trip_fee_pct = 2.0 * float(self.fee_rate)
                    reentry_fee_pct = float(self.fee_rate)
                    slippage_pct = float(getattr(self, "fee_slippage_buffer_pct", 0.0) or 0.0)
                    min_profit_after_fees = float(getattr(self, 'min_profit_after_fees', 0.0010))
                    min_net_profit = max(
                        float(getattr(self, "reversal_min_net_edge_pct", 0.0030) or 0.0030),
                        round_trip_fee_pct + reentry_fee_pct + (2.0 * slippage_pct) + min_profit_after_fees,
                    )

                    if profit_pct < min_net_profit:
                        self.last_status = f"Veto: reversal < net edge ({profit_pct:+.2%} < {min_net_profit:.2%})"
                        return

                    # Age check for cooldown
                    if self.min_seconds_before_reversal and age < float(self.min_seconds_before_reversal):
                        self.last_status = f"Veto: reversal cooldown ({int(self.min_seconds_before_reversal)}s)"
                        return

                    same_side_cooldown = float(getattr(self, "same_side_reentry_cooldown_seconds", 0) or 0)
                    if same_side_cooldown > 0:
                        last_prof_side = str(getattr(self, "_last_profitable_exit_side", "") or "")
                        last_prof_ts = float(getattr(self, "_last_profitable_exit_ts", 0.0) or 0.0)
                        if last_prof_side and last_prof_ts > 0 and (now - last_prof_ts) < same_side_cooldown:
                            if (last_prof_side == "LONG" and action == "BUY") or (last_prof_side == "SHORT" and action == "SELL"):
                                wait_s = int(max(1.0, same_side_cooldown - (now - last_prof_ts)))
                                self.last_status = f"Veto: post-profit same-side cooldown ({wait_s}s)"
                                return

                    # Confidence/Score check
                    conf = float(signal.get("confidence", 0.0) or 0.0)
                    score = float(signal.get("score", 0.0) or 0.0)
                    if conf < float(self.reversal_min_confidence) or abs(score) < float(self.reversal_min_score):
                        self.last_status = "Veto: reversal weak"
                        return

                    if getattr(self, 'exit_on_reversal_only_in_profit', True):
                        logger.info(f"[PROFIT_BANK] Reversal detected while in profit (+{profit_pct:.2%}). Banking green!")
                        order_side = 'SELL' if current_pos['side'] == 'LONG' else 'BUY'
                        order_resp = self.exchange.create_market_order(self.symbol, order_side, current_pos['amount'], params={'reduceOnly': True})
                        current_price = _order_fill_price(order_resp, current_price)

                        pnl = (current_price - current_pos['entry']) * current_pos['amount'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) * current_pos['amount']
                        profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']
                        fees = (current_pos['amount'] * current_pos['entry'] * self.fee_rate) + (current_pos['amount'] * current_price * self.fee_rate)
                        exit_fee = current_pos['amount'] * current_price * self.fee_rate
                        net_pnl = pnl - fees
                        self._record_closed_trade("REVERSAL_BANK", current_pos['entry'], current_price, pnl, profit_pct * 100, fees)
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, exit_fee, t_type="REVERSAL_BANK")
                        self.active_positions = []
                        self._last_trade_ts = now
                        self._last_closed_side = current_pos['side']
                        self._last_profitable_exit_side = current_pos['side']
                        self._last_profitable_exit_ts = now
                        self._opposite_reset_seen_after_profit = False
                        self._cleanup_trade_orders(self.symbol, current_pos)
                        self._post_close_cleanup_needed = True
                        self.last_status = f"REVERSAL_BANK: {current_pos['side']} @ ${current_price:.5f} Net ${net_pnl:+,.2f}"
                        # Continue into the entry path so the reversal signal can flip the position.
                    else:
                        logger.info(f"[REVERSAL] Flipping {current_pos['side']} to {action} (Profit: {profit_pct:+.2%})")
                        order_side = 'SELL' if current_pos['side'] == 'LONG' else 'BUY'
                        order_resp = self.exchange.create_market_order(self.symbol, order_side, current_pos['amount'], params={'reduceOnly': True})
                        current_price = _order_fill_price(order_resp, current_price)

                        pnl = (current_price - current_pos['entry']) * current_pos['amount'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) * current_pos['amount']
                        profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']
                        fees = (current_pos['amount'] * current_pos['entry'] * self.fee_rate) + (current_pos['amount'] * current_price * self.fee_rate)
                        exit_fee = current_pos['amount'] * current_price * self.fee_rate
                        net_pnl = pnl - fees
                        self._record_closed_trade(
                            "REVERSAL",
                            float(current_pos['entry']),
                            float(current_price),
                            float(pnl),
                            float(pnl) * 100.0 / float(current_pos['entry'] * current_pos['amount']),
                            float(fees),
                        )
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, exit_fee, t_type="REVERSAL")
                        self.active_positions = []
                        self._last_trade_ts = now
                        self._last_closed_side = current_pos['side']
                        self._cleanup_trade_orders(self.symbol, current_pos)
                        self._post_close_cleanup_needed = True

            # DCA / Position Check
            is_dca = False
            if self.active_positions:
                pos = self.active_positions[0]
                dca_enabled = getattr(self, 'dca_enabled', False)
                dca_steps = int(pos.get('dca_steps', 0))
                max_dca = int(getattr(self, 'dca_max_steps', 0))
                dist_pct = float(getattr(self, 'dca_distance_pct', 0.01))

                # Check if we should DCA (Must be same direction and meet distance)
                correct_side = (pos['side'] == 'LONG' and action == 'BUY') or (pos['side'] == 'SHORT' and action == 'SELL')
                pnl_pct = (current_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - current_price) / pos['entry']

                if dca_enabled and correct_side and dca_steps < max_dca and pnl_pct <= -dist_pct:
                    is_dca = True
                    logger.info(f"DCA TRIGGERED: Step {dca_steps+1}/{max_dca} (PnL: {pnl_pct:.2%})")
                else:
                    if len(self.active_positions) >= self.max_open_positions:
                        if not getattr(self, 'dca_enabled', False):
                            self._cancel_non_reduce_open_orders(self.symbol)
                        self.last_status = f"In Trade: {pos['side']} (PnL {pnl_pct:+.2%})"
                        return

            same_side_veto = self._same_side_reentry_veto(signal, action, now)
            if same_side_veto:
                self.last_status = same_side_veto
                return

            # 2. Position Entry
            balance = self.get_portfolio_value(current_price)
            if balance <= 0:
                self.last_status = "No equity available (keys/balance)"
                return
            free_balance = self._fetch_free_usdt()
            if free_balance <= 0:
                self.last_status = f"No free margin available (equity ${balance:,.2f})"
                return
            available_balance = min(balance, free_balance)
            if available_balance <= 0:
                self.last_status = "No free margin available"
                return

            # Determine trade size
            dca_enabled = getattr(self, 'dca_enabled', False)
            configured_trade_usdt = float(getattr(self, 'fixed_trade_usdt', 0.0) or 0.0)
            balance_based_cap = available_balance * 0.60
            if is_dca:
                trade_usdt = min(balance_based_cap, available_balance * 0.20)
            elif dca_enabled:
                trade_usdt = min(balance_based_cap, available_balance * 0.40)
            else:
                trade_usdt = balance_based_cap
            if configured_trade_usdt > 0:
                trade_usdt = min(configured_trade_usdt, available_balance * 0.90)

            # Use dynamic leverage based on signal confidence
            confidence = float(signal.get('confidence', 0.5) or 0.5)
            score = float(signal.get('score', 0.5) or 0.5)
            atr_pct = signal.get('atr_pct')  # Get ATR from signal if available
            current_leverage = self.calculate_dynamic_leverage(confidence, score, atr_pct=atr_pct)
            effective_leverage = max(1.0, float(current_leverage))

            max_notional = available_balance * effective_leverage * 0.80
            if trade_usdt > max_notional:
                trade_usdt = max_notional

            if trade_usdt < 5.0:
                self.last_status = f"Equity too low: ${available_balance:,.2f}"
                return

            # CRITICAL: Tell Binance to actually use this leverage, otherwise it uses 1x default and fails
            leverage_set = True
            try:
                exchange_leverage = max(1, int(round(effective_leverage)))
                self.exchange.set_leverage(exchange_leverage, self.symbol)
                effective_leverage = float(exchange_leverage)
            except Exception as e:
                leverage_set = False
                logger.warning(f"Could not set leverage to {int(round(effective_leverage))}x: {e}")
            if not leverage_set:
                self.last_status = "Entry blocked: leverage not confirmed"
                return

            amount = (trade_usdt * effective_leverage) / current_price
            trade_id = self._next_trade_id
            self._next_trade_id += 1

            leverage_info = f" (conf={confidence:.1%}→{effective_leverage:.1f}x)" if self.dynamic_leverage_enabled else ""
            base_asset = str(self.symbol).split('/')[0] if '/' in str(self.symbol) else "units"
            logger.info(
                "Binance Futures %s: %s %s @ %.5f%s | bal=%.2f free=%.2f notional=%.2f",
                action,
                f"{amount:.6f}",
                base_asset,
                current_price,
                leverage_info,
                balance,
                free_balance,
                trade_usdt,
            )
            side = 'BUY' if action == "BUY" else 'SELL'

            # PURGE OLD ORDERS: before a fresh position, remove stale pending
            # entries and reduce-only SL/TP/TS orders from the previous position.
            if not self.active_positions and not is_dca:
                try:
                    self._cleanup_trade_orders(self.symbol)
                except Exception as e:
                    logger.debug(f"Pre-entry stale order cleanup skipped: {e}")
            else:
                try:
                    self._cancel_non_reduce_open_orders(self.symbol)
                except Exception as e:
                    logger.debug(f"Pre-entry non-reduce cleanup skipped: {e}")

            try:
                use_limit = bool(getattr(self, 'use_limit_orders', False))
                real_entry_price = current_price

                if use_limit:
                    resting_price = float(signal.get("resting_entry_price", 0.0) or 0.0)
                    if resting_price > 0:
                        limit_price = resting_price
                    else:
                        limit_price = _maker_entry_price(self.exchange, self.symbol, side, float(signal.get('entry', current_price) or current_price))
                    # Round amount and price to Binance specifications
                    amount_str = self.exchange.amount_to_precision(self.symbol, amount)
                    price_str = self.exchange.price_to_precision(self.symbol, limit_price)

                    order_resp = self.exchange.create_order(
                        symbol=self.symbol,
                        type='LIMIT',
                        side=side,
                        amount=float(amount_str),
                        price=float(price_str),
                        params={'timeInForce': 'GTX', 'postOnly': True}
                    )
                    # Smart SL calculation
                    pos_side = 'LONG' if action == "BUY" else "SHORT"
                    entry_price = float(price_str)
                    support = float(signal.get("structure_support", 0.0) or 0.0)
                    resistance = float(signal.get("structure_resistance", 0.0) or 0.0)
                    atr = float(signal.get("atr", getattr(self, "_current_atr_pct", 0.0) * entry_price) or 0.0)
                    proposed_sl = _smart_sl_price(pos_side, entry_price, support, resistance, atr, atr_multiplier=1.5)
                    logger.info(f"[SMART_SL] {pos_side} entry {entry_price:.2f}: support={support:.2f} resistance={resistance:.2f} ATR={atr:.4f} → SL {proposed_sl:.2f}")

                    self.pending_entry = {
                        'order_id': str(order_resp.get('id') or (order_resp.get('info', {}) or {}).get('orderId') or ''),
                        'action': action,
                        'trade_id': trade_id,
                        'price': float(price_str),
                        'amount': float(amount_str),
                        'trade_usdt': float(trade_usdt),
                        'effective_leverage': float(effective_leverage),
                        'ts': now,
                        'sl': _safe_initial_sl_price(
                            pos_side,
                            entry_price,
                            proposed_sl,
                            getattr(self, 'default_sl_pct', 0.0030),
                            signal.get('pivot_classic'),
                            max_sl_pct=getattr(self, 'max_structural_sl_pct', 0.0120),
                        ),
                        'tp': signal.get('tp'),
                        'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                        'score': float(signal.get("score", 0.0) or 0.0),
                        'confidence': float(signal.get("confidence", 0.0) or 0.0),
                        'reason': str(signal.get("reason", "") or ""),
                        'structure_support': signal.get('structure_support'),
                        'structure_resistance': signal.get('structure_resistance'),
                        'pivot_classic': signal.get('pivot_classic'),
                        'ttl_seconds': float(signal.get("pending_entry_ttl_seconds", getattr(self, "pending_entry_ttl_seconds", 20)) or getattr(self, "pending_entry_ttl_seconds", 20)),
                        'resting_entry': bool(resting_price > 0),
                        'atr': atr,
                    }
                    self.last_status = f"{'Resting' if resting_price > 0 else 'Maker'} entry placed @ {price_str}"
                    return
                else:
                    order_resp = self.exchange.create_market_order(self.symbol, side, amount)
                    real_entry_price = _order_fill_price(order_resp, current_price)

                    self.last_status = f"Market filled: {action} {amount:.0f} @ {real_entry_price:.5f}"
            except Exception as e:
                logger.error(f"Binance Entry Error: {e}")
                self.last_status = f"Entry Error: {str(e)[:40]}"
                err_text = str(e).lower()
                if "-2019" in err_text or "margin is insufficient" in err_text:
                    self._entry_block_until_ts = time.time() + 30.0
                    self.last_status = "Entry blocked: insufficient margin (30s)"
                elif "-5022" in err_text or "post only" in err_text:
                    self._entry_block_until_ts = time.time() + 10.0
                    self.last_status = "Entry blocked: post-only reject (10s)"
                return

            if is_dca and self.active_positions:
                pos = self.active_positions[0]
                old_notional = pos['entry'] * pos['amount']
                new_notional = real_entry_price * amount
                total_amount = pos['amount'] + amount
                new_avg = (old_notional + new_notional) / total_amount

                pos['entry'] = new_avg
                pos['amount'] = total_amount
                pos['dca_steps'] = pos.get('dca_steps', 0) + 1
                logger.info(f"DCA SUCCESS: New Average ${new_avg:.5f}, Amount {total_amount:.0f}")
                self.last_status = f"DCA Step {pos['dca_steps']} filled"
                return

            # The bot uses local runner logic, backed by exchange-side safety orders.
            pos_side = 'LONG' if action == "BUY" else "SHORT"
            # Smart SL calculation
            support = float(signal.get("structure_support", 0.0) or 0.0)
            resistance = float(signal.get("structure_resistance", 0.0) or 0.0)
            atr = float(signal.get("atr", getattr(self, "_current_atr_pct", 0.0) * real_entry_price) or 0.0)
            proposed_sl = _smart_sl_price(pos_side, real_entry_price, support, resistance, atr, atr_multiplier=1.5)
            logger.info(f"[SMART_SL] {pos_side} entry {real_entry_price:.2f}: support={support:.2f} resistance={resistance:.2f} ATR={atr:.4f} → SL {proposed_sl:.2f}")

            sl_price = _safe_initial_sl_price(
                pos_side,
                real_entry_price,
                proposed_sl,
                getattr(self, 'default_sl_pct', 0.0030),
                signal.get('pivot_classic'),
                max_sl_pct=getattr(self, 'max_structural_sl_pct', 0.0120),
            )
            logger.info(f"Using internal dynamic stop logic. Initial soft SL set at {sl_price:.4f}")
            try:
                tp_price = float(signal.get('tp')) if signal.get('tp') else None
            except (TypeError, ValueError):
                tp_price = None
            if tp_price:
                tp_price = _runner_emergency_tp_price(
                    pos_side,
                    real_entry_price,
                    tp_price,
                    getattr(self, "scalp_config", {}) or {},
                )
            runner_enabled = bool((getattr(self, "scalp_config", {}) or {}).get("runner_enabled", True))

            sl_dist = abs(real_entry_price - sl_price) / real_entry_price if real_entry_price and sl_price else float(getattr(self, 'default_sl_pct', 0.0030))
            filled_pos = {
                'trade_id': trade_id,
                'side': pos_side,
                'entry': real_entry_price,
                'amount': amount,
                'trade_usdt': float(trade_usdt),
                'effective_leverage': float(effective_leverage),
                'entry_ts': now,
                'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                'highest_price': current_price if action == "BUY" else 0,
                'lowest_price': current_price if action == "SELL" else 0,
                'highest_profit_pct': 0.0,
                'sl_pct_dist': sl_dist,
                'fee_rate': float(self.fee_rate),
                'min_profit_after_fees': float(getattr(self, 'min_profit_after_fees', 0.0001)),
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
                'fixed_take_profit_enabled': not runner_enabled,
                'structure_support': signal.get('structure_support'),
                'structure_resistance': signal.get('structure_resistance'),
                'native_trailing_activation_pct': float(getattr(self, 'profit_trailing_activation_pct', getattr(self, 'break_even_trigger_pct', 0.0020))),
            }
            self.active_positions.append(filled_pos)
            self._ensure_exchange_stop_loss(filled_pos)
            self._ensure_exchange_take_profit(filled_pos)

            # NATIVE BINANCE TRAILING STOP: Place the official order if enabled
            if getattr(self, 'use_native_trailing_stop', False):
                self._ensure_native_trailing_stop(filled_pos)
            self.trade_count += 1
            self._last_trade_ts = now
            entry_fee = amount * real_entry_price * self.fee_rate
            self._log_trade(
                trade_id,
                "ENTRY",
                action,
                real_entry_price,
                amount,
                fees=entry_fee,
                score=float(signal.get("score", 0.0) or 0.0),
                confidence=float(signal.get("confidence", 0.0) or 0.0),
                reason=str(signal.get("reason", "") or ""),
            )
        except Exception as e:
            logger.error(f"Placement Error: {e}")
            self.last_status = f"Placement error: {e}"

    def close_all_positions(self, symbol: str, reason: str = "close_all_positions"):
        """Liquidate everything on shutdown with Deep Trace and Global Wipe."""
        # Ensure we have the right symbol formats
        raw_symbol = symbol or self.symbol
        target_symbol = _normalize_futures_symbol(raw_symbol)
        target_id = _market_id_from_symbol(target_symbol).upper()
        logger.warning(f"[LIQUIDATE] close_all_positions reason={reason} target={target_symbol}")

        print(f"\n[SHUTDOWN] Starting Deep Trace Cleanup...")
        print(f"  - Target Symbol: {target_symbol}")
        print(f"  - Target ID: {target_id}")

        # First pass: clear tracked local state and any known trade orders.
        try:
            self._cleanup_trade_orders(target_symbol)
        except Exception:
            pass

        for attempt in range(2):
            try:
                # 1. Try CCXT Unified Cancel
                print(f"  - Attempting Unified Cancel (Attempt {attempt+1})...")
                try:
                    self.exchange.cancel_all_orders(target_symbol)
                except Exception as e:
                    print(f"    ! Unified Cancel Error: {e}")

                # 2. Try Direct Binance API Cancel (Ruthless)
                print(f"  - Attempting Direct fapiPrivateDeleteAllOpenOrders...")
                try:
                    self.exchange.fapiPrivateDeleteAllOpenOrders({'symbol': target_id})
                    print("    + Direct API Cancel SUCCESS")
                except Exception as e:
                    print(f"    ! Direct API Cancel Error: {e}")

                time.sleep(1.0)

                # 3. Individual Sweep (Fetch whatever is still alive)
                orders = self.exchange.fetch_open_orders(target_symbol)
                if orders:
                    print(f"  - Found {len(orders)} persistent orders. Killing them individually...")
                    for o in orders:
                        try:
                            self.exchange.cancel_order(o['id'], target_symbol)
                            print(f"    + Killed order {o['id']}")
                        except:
                            pass

                # 4. Position Liquidation
                print(f"  - Checking for active positions...")
                balance = self.exchange.fetch_balance()
                if 'info' in balance and 'positions' in balance['info']:
                    for p in balance['info']['positions']:
                        p_id = str(p['symbol']).upper()
                        if p_id == target_id:
                            amt = float(p['positionAmt'])
                            if abs(amt) > 0.0:
                                side = 'SELL' if amt > 0 else 'BUY'
                                print(f"  - FOUND POSITION: {amt} units. LIQUIDATING NOW...")
                                logger.warning(
                                    f"[LIQUIDATE] Sending reduce-only MARKET {side} {abs(amt):.8f} {target_symbol} "
                                    f"reason={reason}"
                                )
                                self.exchange.create_market_order(target_symbol, side, abs(amt), params={'reduceOnly': True})
                                print(f"    + Liquidation order sent.")
                                try:
                                    self._cleanup_trade_orders(target_symbol)
                                except Exception:
                                    pass
                                try:
                                    self._cancel_non_reduce_open_orders(target_symbol)
                                except Exception:
                                    pass
                                time.sleep(0.5)

                # Final Verification
                remaining = self.exchange.fetch_open_orders(target_symbol)
                if not remaining:
                    print("  - VERIFIED: All orders cleared.")
                    break
            except Exception as e:
                print(f"  [!] Cleanup Error: {e}")
                time.sleep(1.0)

        # Final sweep: if anything was recreated during the close, wipe it again.
        try:
            self._cleanup_trade_orders(target_symbol)
            self._cancel_non_reduce_open_orders(target_symbol)
            self.exchange.cancel_all_orders(target_symbol)
        except Exception:
            pass

        self.active_positions = []
        self.pending_entry = None
        self.pending_exit = None
        print("[SHUTDOWN] CLEANUP COMPLETE. Account is FLAT.")

