from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from ..base import (
    _compute_trailing_stop,
    _exchange_flag_true,
    _maker_entry_price,
    _order_fill_price,
    _order_id,
    _order_trigger_price,
    _order_type,
    _realized_exit_type,
)

logger = logging.getLogger(__name__)

class ExitMixin:
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

