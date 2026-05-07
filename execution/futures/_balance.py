from __future__ import annotations

import logging

from ..base import _quote_asset_from_symbol

logger = logging.getLogger(__name__)

class BalanceMixin:
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

