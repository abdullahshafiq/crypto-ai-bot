from __future__ import annotations

import logging
import os

from ..base import (
    _exchange_flag_true,
    _market_id_from_symbol,
    _order_id,
    _order_type,
)

logger = logging.getLogger(__name__)

class OrderMixin:
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

