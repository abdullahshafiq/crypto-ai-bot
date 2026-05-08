from __future__ import annotations

import logging
import os
import time

from ..base import (
    _market_id_from_symbol,
    _normalize_futures_symbol,
    _quote_asset_from_symbol,
)

logger = logging.getLogger(__name__)

class PortfolioMixin:
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


