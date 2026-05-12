from __future__ import annotations

import logging
import os
import time

from ..base import (
    _maker_entry_price,
    _market_id_from_symbol,
    _normalize_futures_symbol,
    _order_fill_price,
    _runner_emergency_tp_price,
    _safe_initial_sl_price,
    _smart_sl_price,
)

logger = logging.getLogger(__name__)

class EntryMixin:
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
                                existing['trailing_tp_keep_ratio'] = float(existing.get('trailing_tp_keep_ratio', 0.0) or 0.0)
                                existing['trailing_tp_peak_pct'] = float(existing.get('trailing_tp_peak_pct', 0.0) or 0.0)
                                existing['trailing_tp_floor_pct'] = float(existing.get('trailing_tp_floor_pct', 0.0) or 0.0)
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
                        # Capture signal diagnostics for logging and post-trade analysis
                        signal_reason = str(signal.get('reason', '') or '')[:180]
                        entry_mode = str(signal.get('entry_mode', 'UNKNOWN') or 'UNKNOWN')
                        signal_score = float(signal.get('score', 0.0) or 0.0)
                        signal_confidence = float(signal.get('confidence', 0.0) or 0.0)

                        filled_pos = {
                            'trade_id': trade_id,
                            'side': pos_side,
                            'entry': fill_price,
                            'amount': amount,
                            'trade_usdt': float(pending_entry.get('trade_usdt', 0.0) or 0.0),
                            'effective_leverage': float(pending_entry.get('effective_leverage', getattr(self, 'leverage', 1.0)) or getattr(self, 'leverage', 1.0)),
                            'entry_ts': now,
                            'hold_until_ts': float(pending_entry.get("hold_until_ts", 0.0) or 0.0),
                            'signal_reason': signal_reason,
                            'entry_mode': entry_mode,
                            'signal_score': signal_score,
                            'signal_confidence': signal_confidence,
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
                            'trailing_tp_keep_ratio': 0.0,
                            'trailing_tp_peak_pct': 0.0,
                            'trailing_tp_floor_pct': 0.0,
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
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, exit_fee, t_type="REVERSAL_BANK", signal_reason=str(current_pos.get("signal_reason", "") or ""), entry_mode=str(current_pos.get("entry_mode", "") or ""))
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
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, exit_fee, t_type="REVERSAL", signal_reason=str(current_pos.get("signal_reason", "") or ""), entry_mode=str(current_pos.get("entry_mode", "") or ""))
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
            # Live sizing: keep a 10% reserve and use the rest as the margin budget.
            balance_based_cap = available_balance * 0.90
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
                'trailing_tp_keep_ratio': 0.0,
                'trailing_tp_peak_pct': 0.0,
                'trailing_tp_floor_pct': 0.0,
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
                signal_reason=str(signal.get("reason", "") or "")[:180],
                entry_mode=str(signal.get("entry_mode", "") or ""),
            )
        except Exception as e:
            logger.error(f"Placement Error: {e}")
            self.last_status = f"Placement error: {e}"

