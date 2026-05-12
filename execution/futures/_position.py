from __future__ import annotations

import logging
import os
import time

from ..base import (
    _compute_trailing_stop,
    _exchange_flag_true,
    _market_id_from_symbol,
    _normalize_futures_symbol,
    _order_fill_price,
    _order_id,
    _realized_exit_type,
    _runner_emergency_tp_price,
    _safe_initial_sl_price,
    _safe_tp_price,
    _trailing_tp_hit,
)

logger = logging.getLogger(__name__)

class PositionManagerMixin:
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
                        self._log_trade(pos.get("trade_id", 0), "EXIT", pending_exit.get("side", "SELL"), fill_price, amount, pnl, exit_fee, t_type=exit_type, signal_reason=str(pos.get("signal_reason", "") or ""), entry_mode=str(pos.get("entry_mode", "") or ""))
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
                            signal_reason=str(pos.get("signal_reason", "") or ""),
                            entry_mode=str(pos.get("entry_mode", "") or ""),
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
                        self._log_trade(trade_id, "EXIT", order_side, fill_price, amount, pnl, exit_fee, t_type=exit_type, signal_reason=str(pos.get("signal_reason", "") or ""), entry_mode=str(pos.get("entry_mode", "") or ""))
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
                        signal_reason="",
                        entry_mode="ADOPTED",
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

