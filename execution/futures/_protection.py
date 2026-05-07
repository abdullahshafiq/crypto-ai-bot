from __future__ import annotations

import logging
import os
import time

from ..base import (
    _market_id_from_symbol,
    _order_id,
    _order_trigger_price,
)

logger = logging.getLogger(__name__)

class ProtectionMixin:
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

