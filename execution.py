import ccxt
import time
import csv
import os
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

TRADE_LOG_FILE = "trade_log_futures.csv"
TRADE_LOG_HEADER = [
    "timestamp",
    "trade_id",
    "event",
    "side",
    "price",
    "amount",
    "pnl",
    "fees",
    "score",
    "confidence",
    "reason",
    "type",
]

def _normalize_futures_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip()
    if not symbol:
        return symbol
    if ":" in symbol:
        return symbol
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        quote = quote.split(":")[0]
        return f"{base}/{quote}:{quote}"
    return symbol


def _normalize_spot_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip()
    if not symbol:
        return symbol
    if ":" in symbol:
        base, quote = symbol.split(":", 1)[0].split("/", 1)
        return f"{base}/{quote}"
    return symbol


def _market_id_from_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace(":USDT", "").replace(":USDC", "")


def _quote_asset_from_symbol(symbol: str) -> str:
    symbol = _normalize_futures_symbol(symbol or "DOGE/USDT")
    if "/" not in symbol:
        return "USDT"
    quote = symbol.split("/", 1)[1].split(":", 1)[0].upper()
    return quote or "USDT"


def _order_fill_price(order: dict, fallback: float) -> float:
    if not isinstance(order, dict):
        return float(fallback)
    for key in ("average", "price"):
        value = order.get(key)
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def _maker_entry_price(exchange, symbol: str, side: str, fallback: float) -> float:
    """
    Place maker entries close to the book without crossing it.
    BUY: near best bid
    SELL: near best ask
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=5) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        best_bid = float(bids[0][0]) if bids else float(fallback)
        best_ask = float(asks[0][0]) if asks else float(fallback)
        tick = max(best_bid * 0.0001, fallback * 0.00005, 0.0000001)
        side_u = str(side or "").upper()

        if side_u == "BUY":
            price = min(best_bid + tick, best_ask - tick) if best_ask > best_bid else best_bid + tick
            return float(price)

        price = max(best_ask - tick, best_bid + tick) if best_ask > best_bid else best_ask - tick
        return float(price)
    except Exception:
        return float(fallback)


def _realized_exit_type(exit_type: str, net_pnl: float) -> str:
    base = str(exit_type or "EXIT").upper()
    if base in {"TRAIL_WIN", "TRAIL_SL"}:
        return "TRAIL_WIN" if float(net_pnl) > 0 else "TRAIL_SL"
    if base in {"TAKE_PROFIT", "SCALP_EXIT", "REVERSAL_BANK", "REVERSAL", "REVERSAL_WIN", "REVERSAL_CUT"}:
        return base if float(net_pnl) > 0 else f"{base}_LOSS"
    return base if float(net_pnl) > 0 else f"{base}_LOSS"


def _exchange_flag_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _default_sl_price(side: str, entry: float, sl_pct: float) -> float:
    side = str(side or "").upper()
    entry = float(entry)
    sl_pct = abs(float(sl_pct or 0.0030))
    return entry * (1 - sl_pct) if side in {"LONG", "BUY"} else entry * (1 + sl_pct)


def _safe_initial_sl_price(side: str, entry: float, proposed_sl, sl_pct: float) -> float:
    side_u = str(side or "").upper()
    entry = float(entry)
    default_sl = _default_sl_price(side_u, entry, sl_pct)
    min_dist_pct = abs(float(sl_pct or 0.0030))
    try:
        sl = float(proposed_sl)
    except (TypeError, ValueError):
        return default_sl

    if entry <= 0 or sl <= 0:
        return default_sl

    if side_u in {"LONG", "BUY"}:
        if sl >= entry:
            return default_sl
        if ((entry - sl) / entry) < min_dist_pct:
            return default_sl
    else:
        if sl <= entry:
            return default_sl
        if ((sl - entry) / entry) < min_dist_pct:
            return default_sl

    return sl


def _compute_trailing_stop(pos: dict, current_price: float, current_psar: float = None) -> float:
    entry = float(pos.get("entry", current_price) or current_price)
    side = str(pos.get("side", "LONG"))
    base_dist = float(pos.get("sl_pct_dist", 0.005) or 0.005)
    break_even_trigger = float(pos.get("break_even_trigger_pct", 0.0010) or 0.0010)
    break_even_buffer = float(pos.get("break_even_buffer_pct", 0.0002) or 0.0002)
    trail_tighten_1 = float(pos.get("trail_tighten_1_pct", 0.0030) or 0.0030)
    trail_tighten_2 = float(pos.get("trail_tighten_2_pct", 0.0060) or 0.0060)
    fee_rate = float(pos.get("fee_rate", 0.0004) or 0.0004)
    min_profit_after_fees = float(pos.get("min_profit_after_fees", 0.0002) or 0.0002)

    shadow_trigger = 0.0015  # Enter Shadow Mode at 0.15% profit
    shadow_gap = 0.0008      # Hug price at 0.08% distance
    
    if side == "LONG":
        best_price = float(pos.get("highest_price", entry) or entry)
        profit_pct = (best_price - entry) / entry if entry else 0.0
        
        # Start with standard percentage trail
        trail_sl = best_price * (1 - base_dist)
        
        # 1. STRUCTURAL OVERRIDE
        support = pos.get('structure_support')
        if support and float(support) < current_price:
            trail_sl = max(trail_sl, float(support))
            
        # 2. BREAK-EVEN LOCK
        if profit_pct >= break_even_trigger:
            min_sl = entry * (1 + (2.0 * fee_rate) + min_profit_after_fees)
            trail_sl = max(trail_sl, min_sl)
            
        # 3. SHADOW CHASE (MAX PROFIT)
        if profit_pct >= shadow_trigger:
            shadow_sl = best_price * (1 - shadow_gap)
            trail_sl = max(trail_sl, shadow_sl)
            
        return min(trail_sl, current_price * (1 - 0.0005))

    best_price = float(pos.get("lowest_price", entry) or entry)
    profit_pct = (entry - best_price) / entry if entry else 0.0
    
    # Start with standard percentage trail
    trail_sl = best_price * (1 + base_dist)
    
    # 1. STRUCTURAL OVERRIDE
    resistance = pos.get('structure_resistance')
    if resistance and float(resistance) > current_price:
        trail_sl = min(trail_sl, float(resistance))
    
    # 2. BREAK-EVEN LOCK
    if profit_pct >= break_even_trigger:
        max_sl = entry * (1 - (2.0 * fee_rate) - min_profit_after_fees)
        trail_sl = min(trail_sl, max_sl)
        
    # 3. SHADOW CHASE (MAX PROFIT)
    if profit_pct >= shadow_trigger:
        shadow_sl = best_price * (1 + shadow_gap)
        trail_sl = min(trail_sl, shadow_sl)
        
    return max(trail_sl, current_price * (1 + 0.0005))

class BinanceFuturesExecution:
    def __init__(self, api_key: str, api_secret: str, symbol: str = "SOL/USDC:USDC", leverage: int = 5, max_closed_trades: int = 5000, is_demo: bool = True):
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
        self.break_even_trigger_pct = 0.0010
        self.break_even_buffer_pct = 0.0002
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self.default_sl_pct = 0.0030
        self.exit_on_reversal_only_in_profit = True
        self._last_trade_ts = 0.0
        self.trade_log_file = TRADE_LOG_FILE
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
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
        self.pending_entry = None
        self.pending_exit = None
        self.pending_entry_ttl_seconds = 20
        self.pending_exit_ttl_seconds = 20
        self._entry_block_until_ts = 0.0
        
        self.max_open_positions = 1
        self.min_balance_floor = 0.0
        self.daily_loss_cap_pct = None
        
        self.leverage_score_weight = 0.3
        self.min_profit_after_fees = 0.0002
        self.exit_on_reversal_only_in_profit = True
        self.use_native_trailing_stop = False
        self.trailing_stop_callback = 0.005
        self.default_sl_pct = 0.0030
        self._last_position_sync_ok = False
        self._last_flat_order_cleanup_ts = 0.0
        self._entry_block_until_ts = 0.0
        self.scalp_config = {
            'tp_pct': 0.0040,
            'min_hold_seconds': 20,
            'fade_trigger_pct': 0.0060,
            'fade_exit_pct': 0.0030
        }

        self.last_status = "INIT"
        self._last_price = None
        
        self.initial_balance = 0.0  
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

        self._init_trade_log()
        logger.info("Binance Futures Testnet Initialized.")

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
            current_atr = atr_pct if atr_pct is not None else getattr(self, '_current_atr_pct', 0.02)
            atr_reference = getattr(self, 'atr_reference_pct', 0.02)
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

    def _record_closed_trade(self, t_type: str, entry: float, exit_price: float, pnl: float, pnl_pct: float, fees: float):
        net_pnl = float(pnl) - float(fees)
        self.closed_trades.append({
            "type": t_type,
            "entry": float(entry),
            "exit": float(exit_price),
            "pnl": float(net_pnl),
            "pnl_pct": float(pnl_pct),
            "fees": float(fees),
        })
        self.stats_trades += 1
        if net_pnl > 0:
            self.stats_wins += 1
        else:
            self.stats_losses += 1
        self.stats_gross += float(pnl)
        self.stats_fees += float(fees)

    def _fetch_free_usdt(self):
        try:
            balance = self.exchange.fetch_balance()
            target_asset = _quote_asset_from_symbol(getattr(self, 'symbol', 'DOGE/USDT'))
            # 1. Standard CCXT
            free = float(balance.get('free', {}).get(target_asset, 0.0))
            if free > 0: return free
            
            # 2. Try total
            total = float(balance.get('total', {}).get(target_asset, 0.0))
            if total > 0: return total
            
            # 3. Deep dive into raw 'info' from Binance API
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
                            return float(asset.get('availableBalance', 0.0))
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
        if pos.get('native_trailing_order_id'):
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

    def _cleanup_trade_orders(self, symbol: str = None, pos: dict = None):
        """
        Cancel all open trade-related orders and clear local pending order state.
        """
        target_symbol = symbol or self.symbol
        try:
            self.exchange.cancel_all_orders(target_symbol)
        except Exception as e:
            logger.debug(f"Cleanup cancel_all_orders skipped: {e}")

        if isinstance(pos, dict):
            trail_id = str(pos.get('native_trailing_order_id') or '')
            if trail_id:
                try:
                    self.exchange.cancel_order(trail_id, target_symbol)
                except Exception as e:
                    logger.debug(f"Cleanup cancel trailing skipped: {e}")
            pos.pop('native_trailing_order_id', None)
            pos.pop('native_trailing_activation_price', None)
            pos.pop('native_trailing_callback_pct', None)

        if getattr(self, "pending_entry", None):
            self.pending_entry = None
        if getattr(self, "pending_exit", None):
            self.pending_exit = None

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
            # On Futures, total equity is 'total' -> active quote asset.
            total_equity = float(balance.get('total', {}).get(target_asset, 0.0))

            if total_equity == 0 and 'info' in balance:
                info = balance['info']
                if 'assets' in info:
                    for asset in info['assets']:
                        if asset.get('asset') == target_asset:
                            total_equity = float(asset.get('walletBalance', asset.get('marginBalance', 0.0)) or 0.0)
                            break
                if total_equity == 0:
                    total_equity = float(info.get('totalWalletBalance', 0.0) or 0.0)
            
            if not self._initial_price_set and total_equity > 0:
                self.initial_balance = total_equity
                self._initial_price_set = True
                logger.info(f"Initial Session Equity: ${self.initial_balance:,.2f}")
            return total_equity
        except Exception as e:
            logger.error(f"Equity Fetch Error: {e}")
            self.last_status = f"Equity error: {e}"
            return 0.0

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
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type(str(pending_exit.get("exit_type", "TRAIL_WIN") or "TRAIL_WIN"), net_pnl)
                        self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
                        self._log_trade(pos.get("trade_id", 0), "EXIT", pending_exit.get("side", "SELL"), fill_price, amount, net_pnl, fees, t_type=exit_type)
                        self.trade_count += 1
                        self._last_trade_ts = time.time()
                        self._recently_closed_ts = time.time()
                        self._last_closed_side = side
                        self._cleanup_trade_orders(self.symbol, pos)
                        self.active_positions = []
                        self.pending_exit = None
                        self.last_status = f"{exit_type}: {side} @ ${fill_price:.5f} Net ${net_pnl:+,.2f}"
                        return
                elif age < float(getattr(self, "pending_exit_ttl_seconds", 20) or 20):
                    self.last_status = f"Waiting maker exit fill @ ${float(pending_exit.get('price', current_price) or current_price):.5f}"
                    return
                else:
                    if order_id:
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                        except Exception:
                            pass
                    self.pending_exit = None
                    self.last_status = "Maker exit expired; reprice"
                    return
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
                # No position on exchange. Keep pending maker entries alive so they
                # can fill within TTL; cleanup only real stale position/exit state.
                if self.active_positions or getattr(self, "pending_exit", None):
                    logger.info(f"[SYNC] No position on exchange for {self.symbol}, clearing local state.")
                    if self.active_positions:
                        pos = self.active_positions[0]
                        entry = float(pos.get("entry", current_price) or current_price)
                        amount = float(pos.get("amount", 0.0) or 0.0)
                        side = str(pos.get("side", "LONG") or "LONG").upper()
                        trade_id = int(pos.get("trade_id", 0) or 0)
                        pnl = (current_price - entry) * amount if side == "LONG" else (entry - current_price) * amount
                        profit_pct = (current_price - entry) / entry if side == "LONG" else (entry - current_price) / entry
                        fees = (amount * entry * self.fee_rate) + (amount * current_price * self.fee_rate)
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type("EXCHANGE_CLOSED", net_pnl)
                        order_side = "SELL" if side == "LONG" else "BUY"
                        self._record_closed_trade(exit_type, entry, current_price, pnl, profit_pct * 100, fees)
                        self._log_trade(trade_id, "EXIT", order_side, current_price, amount, net_pnl, fees, t_type=exit_type)
                        self.trade_count += 1
                        self._last_trade_ts = time.time()
                        self._recently_closed_ts = time.time()
                        self._last_closed_side = side
                    try:
                        self._cleanup_trade_orders(self.symbol, self.active_positions[0] if self.active_positions else None)
                    except Exception as e:
                        logger.debug(f"[SYNC] Open-order cleanup skipped: {e}")
                elif not getattr(self, "pending_entry", None):
                    now_cleanup = time.time()
                    if now_cleanup - float(getattr(self, "_last_flat_order_cleanup_ts", 0.0) or 0.0) >= 30:
                        try:
                            self.exchange.cancel_all_orders(self.symbol)
                            self._last_flat_order_cleanup_ts = now_cleanup
                        except Exception as e:
                            logger.debug(f"[SYNC] Flat stale-order cleanup skipped: {e}")
                self.active_positions = []
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
                    pending_support = None
                    pending_resistance = None
                    if isinstance(pending_entry, dict):
                        pending_action = str(pending_entry.get("action", "") or "").upper()
                        pending_side = "LONG" if pending_action == "BUY" else ("SHORT" if pending_action == "SELL" else "")
                        if pending_side == exch_side:
                            pending_sl = pending_entry.get("sl")
                            pending_support = pending_entry.get("structure_support")
                            pending_resistance = pending_entry.get("structure_resistance")
                            self.pending_entry = None
                    initial_sl = _safe_initial_sl_price(
                        exch_side,
                        entry_price,
                        pending_sl,
                        getattr(self, 'default_sl_pct', 0.0030),
                    )
                    # ADOPT position: Give safety room to prevent instant stop-outs on reboot
                    initial_sl = _default_sl_price(exch_side, entry_price, float(getattr(self, 'default_sl_pct', 0.0050)))
                    
                    adopted_pos = {
                        'trade_id': 9999, 
                        'side': exch_side,
                        'entry': entry_price,
                        'amount': exch_size,
                        'entry_ts': time.time(),
                        'highest_price': current_price if exch_side == 'LONG' else 0,
                        'lowest_price': current_price if exch_side == 'SHORT' else 0,
                        'highest_profit_pct': 0.0,
                        'sl': initial_sl,
                        'sl_pct_dist': abs(entry_price - initial_sl) / entry_price if entry_price else 0.0050,
                        'fee_rate': getattr(self, 'fee_rate', 0.0004),
                        'min_profit_after_fees': getattr(self, 'min_profit_after_fees', 0.0005),
                        'break_even_trigger_pct': float(getattr(self, 'break_even_trigger_pct', 0.0010)),
                        'break_even_buffer_pct': float(getattr(self, 'break_even_buffer_pct', 0.0002)),
                        'trail_tighten_1_pct': float(getattr(self, 'trail_tighten_1_pct', 0.0030)),
                        'trail_tighten_2_pct': float(getattr(self, 'trail_tighten_2_pct', 0.0060)),
                        'trail_armed': False,
                        'structure_support': pending_support,
                        'structure_resistance': pending_resistance,
                        'native_trailing_activation_pct': float(getattr(self, 'break_even_trigger_pct', 0.0010)),
                    }
                    logger.warning(f"[ADOPT] Found existing {exch_side} position. Adopting with safety SL at {initial_sl:.4f}")
                    self.active_positions.append(adopted_pos)
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
            primary = next((p for p in self.active_positions if int(p.get('trade_id', 9999) or 9999) != 9999), self.active_positions[0])
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
        
        self._last_position_sync_ok = True
        self._last_sync_ts = time.time()
        
        remaining = []
        try:
            for pos in self.active_positions:
                closed = False
                entry = pos['entry']
                side = pos['side']
                amount = pos['amount']
                trade_id = pos.get("trade_id", 0)

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

                    if not closed and tp_price > 0 and current_price >= tp_price and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TAKE_PROFIT"
                    
                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            if pos.get("tp_target"):
                                expected_tp_pct = abs(float(pos["tp_target"]) - entry) / entry
                                if expected_tp_pct > 0.001:
                                    tp_pct = expected_tp_pct
                                    
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_FADE"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TTL_EXIT"

                    # DIAMOND HANDS: Trailing stop now follows market structure.
                    new_sl = _compute_trailing_stop(pos, current_price)
                    if new_sl > float(pos['sl']):
                        old_sl = float(pos['sl'])
                        pos['sl'] = new_sl
                        logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Support Lock)")
                    
                    if current_price <= pos['sl']:
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

                    if not closed and tp_price > 0 and current_price <= tp_price and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TAKE_PROFIT"

                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            if pos.get("tp_target"):
                                expected_tp_pct = abs(float(pos["tp_target"]) - entry) / entry
                                if expected_tp_pct > 0.001:
                                    tp_pct = expected_tp_pct
                                    
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_FADE"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct >= min_net_profit:
                        closed = True
                        exit_type = "TTL_EXIT"

                    # DIAMOND HANDS: Trailing stop now follows market structure.
                    new_sl = _compute_trailing_stop(pos, current_price)
                    if new_sl < float(pos['sl']):
                        old_sl = float(pos['sl'])
                        pos['sl'] = new_sl
                        logger.info(f"[STRUCTURAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f} (Resistance Lock)")

                    if current_price >= pos['sl']:
                        closed = True
                        exit_type = "STOP_LOSS"
                
                if closed:
                    profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
                    logger.info(f"[FUTURES] Exit {side} @ {current_price:.2f}")

                    order_side = 'SELL' if side == 'LONG' else 'BUY'
                    self._cleanup_trade_orders(self.symbol, pos)

                    stop_like_exit = exit_type in {"TRAIL_WIN", "TRAIL_SL", "TAKE_PROFIT", "SCALP_EXIT", "TTL_EXIT"}
                    if getattr(self, 'use_limit_orders', False) and not stop_like_exit:
                        try:
                            exit_limit = _maker_entry_price(self.exchange, self.symbol, order_side, float(current_price))
                            price_s = self.exchange.price_to_precision(self.symbol, exit_limit)
                            order_resp = self.exchange.create_order(
                                symbol=self.symbol,
                                type='LIMIT',
                                side=order_side,
                                amount=amount,
                                price=float(price_s),
                                params={'reduceOnly': True, 'postOnly': True, 'timeInForce': 'GTX'}
                            )
                            self.pending_exit = {
                                'order_id': str(order_resp.get('id') or (order_resp.get('info', {}) or {}).get('orderId') or ''),
                                'ts': time.time(),
                                'price': float(price_s),
                                'amount': amount,
                                'side': order_side,
                                'exit_type': exit_type,
                            }
                            self.last_status = f"Maker exit placed @ {price_s}"
                        except Exception as e:
                            logger.warning(f"Maker exit placement failed: {e}")
                            self.last_status = "Maker exit failed; still tracking position"
                        remaining.append(pos)
                        continue
                    if stop_like_exit:
                        logger.info("[EXIT] Using market reduce-only close for stop/trail protection.")
                    try:
                        order_resp = self.exchange.create_market_order(self.symbol, order_side, amount, params={'reduceOnly': True})
                        current_price = _order_fill_price(order_resp, current_price)
                        logger.info(f"[MARKET_EXIT] {order_side} filled at {current_price:.5f}")
                    except Exception as e:
                        logger.warning(f"Market exit failed; keeping position active: {e}")
                        self.last_status = "Market exit failed; still tracking position"
                        remaining.append(pos)
                        continue

                    pnl = (current_price - entry) * amount if side == 'LONG' else (entry - current_price) * amount
                    profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
                    fees = (amount * entry * self.fee_rate) + (amount * current_price * self.fee_rate)
                    net_pnl = pnl - fees
                    exit_type = _realized_exit_type(exit_type or "TRAIL_WIN", net_pnl)

                    self._record_closed_trade(exit_type, entry, current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(trade_id, "EXIT", order_side, current_price, amount, net_pnl, fees, t_type=exit_type)
                    self.trade_count += 1
                    self._last_trade_ts = time.time()
                    self._recently_closed_ts = time.time()
                    self._last_closed_side = side
                    self._cleanup_trade_orders(self.symbol, pos)
                    self.last_status = f"{exit_type}: {side} @ ${current_price:.5f} Net ${net_pnl:+,.2f}"
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
                        default_sl = _default_sl_price(pos_side, fill_price, getattr(self, 'default_sl_pct', 0.0030))
                        sl_price = _safe_initial_sl_price(
                            pos_side,
                            fill_price,
                            pending_entry.get("sl", default_sl),
                            getattr(self, 'default_sl_pct', 0.0030),
                        )
                        try:
                            tp_price = float(pending_entry.get("tp")) if pending_entry.get("tp") else None
                        except (TypeError, ValueError):
                            tp_price = None
                        if self.active_positions:
                            existing = self.active_positions[0]
                            if str(existing.get('side', '')).upper() == pos_side:
                                existing['trade_id'] = existing.get('trade_id') or trade_id
                                existing['entry'] = fill_price
                                existing['amount'] = amount
                                existing['sl'] = sl_price
                                existing['sl_pct_dist'] = abs(fill_price - sl_price) / fill_price if fill_price else existing.get('sl_pct_dist', 0.005)
                                existing['tp_price'] = tp_price
                                existing['structure_support'] = pending_entry.get('structure_support')
                                existing['structure_resistance'] = pending_entry.get('structure_resistance')
                                existing['trail_armed'] = bool(existing.get('trail_armed', False))
                                self._ensure_native_trailing_stop(existing)
                                self.pending_entry = None
                                self.last_status = f"Maker entry synced @ {fill_price:.5f}"
                                return
                        filled_pos = {
                            'trade_id': trade_id,
                            'side': pos_side,
                            'entry': fill_price,
                            'amount': amount,
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
                            'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                            'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
                            'trail_armed': False,
                            'sl': sl_price,
                            'tp_price': tp_price,
                            'structure_support': pending_entry.get('structure_support'),
                            'structure_resistance': pending_entry.get('structure_resistance'),
                            'native_trailing_activation_pct': float(getattr(self, 'break_even_trigger_pct', 0.0020)),
                        }
                        self.active_positions.append(filled_pos)
                        self._ensure_native_trailing_stop(filled_pos)
                        self.pending_entry = None
                        self.last_status = f"Maker entry filled @ {fill_price:.5f}"
                        return
                    elif age < float(getattr(self, "pending_entry_ttl_seconds", 3) or 3):
                        self.last_status = f"Waiting entry fill @ ${float(pending_entry.get('price', current_price) or current_price):.5f}"
                        return
                    else:
                        # HYBRID SNIPER: Maker entry failed to fill in time. Execute MARKET fallback.
                        if order_id:
                            try:
                                self.exchange.cancel_order(order_id, self.symbol)
                            except Exception:
                                pass
                        
                        logger.warning(f"Maker entry timed out after {int(age)}s. Executing MARKET fallback to ensure entry.")
                        action = pending_entry['action']
                        side = 'BUY' if action == "BUY" else 'SELL'
                        amount = pending_entry['amount']
                        trade_id = pending_entry['trade_id']
                        
                        try:
                            order_resp = self.exchange.create_market_order(self.symbol, side, amount)
                            real_entry_price = _order_fill_price(order_resp, current_price)
                            
                            # Convert pending state to active position
                            pos_side = 'LONG' if action == "BUY" else "SHORT"
                            sl_price = pending_entry.get('sl', current_price * 0.99)
                            
                            filled_pos = {
                                'trade_id': trade_id,
                                'side': pos_side,
                                'entry': real_entry_price,
                                'amount': amount,
                                'entry_ts': time.time(),
                                'hold_until_ts': float(pending_entry.get("hold_until_ts", 0.0) or 0.0),
                                'highest_price': current_price if action == "BUY" else 0,
                                'lowest_price': current_price if action == "SELL" else 0,
                                'highest_profit_pct': 0.0,
                                'sl_pct_dist': abs(real_entry_price - sl_price) / real_entry_price if real_entry_price else 0.005,
                                'fee_rate': float(self.fee_rate),
                                'min_profit_after_fees': float(getattr(self, 'min_profit_after_fees', 0.0001)),
                                'break_even_trigger_pct': float(self.break_even_trigger_pct),
                                'break_even_buffer_pct': float(self.break_even_buffer_pct),
                                'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                                'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
                                'trail_armed': False,
                                'sl': sl_price,
                                'tp_price': pending_entry.get('tp'),
                                'structure_support': pending_entry.get('structure_support'),
                                'structure_resistance': pending_entry.get('structure_resistance'),
                                'native_trailing_activation_pct': float(getattr(self, 'break_even_trigger_pct', 0.0020)),
                            }
                            self.active_positions.append(filled_pos)
                            self._ensure_native_trailing_stop(filled_pos)
                            self.pending_entry = None
                            self.trade_count += 1
                            self._last_trade_ts = time.time()
                            self.last_status = f"Hybrid Market Fill: {action} @ {real_entry_price:.5f}"
                            return
                        except Exception as e:
                            logger.error(f"Hybrid Market Fallback Error: {e}")
                            self.pending_entry = None
                            self.last_status = "Hybrid Fallback Failed"
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
                        net_pnl = pnl - fees
                        self._record_closed_trade("REVERSAL_BANK", current_pos['entry'], current_price, pnl, profit_pct * 100, fees)
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], net_pnl, fees, t_type="REVERSAL_BANK")
                        self.active_positions = []
                        self._last_trade_ts = now
                        self.last_status = f"REVERSAL_BANK: {current_pos['side']} @ ${current_price:.5f} Net ${net_pnl:+,.2f}"
                        # Stays flat as per dynamic bank logic
                    else:
                        logger.info(f"[REVERSAL] Flipping {current_pos['side']} to {action} (Profit: {profit_pct:+.2%})")
                        order_side = 'SELL' if current_pos['side'] == 'LONG' else 'BUY'
                        order_resp = self.exchange.create_market_order(self.symbol, order_side, current_pos['amount'], params={'reduceOnly': True})
                        current_price = _order_fill_price(order_resp, current_price)
                        
                        pnl = (current_price - current_pos['entry']) * current_pos['amount'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) * current_pos['amount']
                        profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']
                        fees = (current_pos['amount'] * current_pos['entry'] * self.fee_rate) + (current_pos['amount'] * current_price * self.fee_rate)
                        net_pnl = pnl - fees
                        self._record_closed_trade(
                            "REVERSAL",
                            float(current_pos['entry']),
                            float(current_price),
                            float(pnl),
                            float(pnl) * 100.0 / float(current_pos['entry'] * current_pos['amount']),
                            float(fees),
                        )
                        self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], net_pnl, fees, t_type="REVERSAL")
                        self.active_positions = []
                        self._last_trade_ts = now

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
                        self.last_status = f"In Trade: {pos['side']} (PnL {pnl_pct:+.2%})"
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

            # Determine trade size conservatively so live does not over-allocate.
            dca_enabled = getattr(self, 'dca_enabled', False)
            configured_trade_usdt = float(getattr(self, 'fixed_trade_usdt', 0.0) or 0.0)
            balance_based_cap = available_balance * 0.15
            if is_dca:
                trade_usdt = min(balance_based_cap, available_balance * 0.20)
            elif dca_enabled:
                trade_usdt = min(balance_based_cap, available_balance * 0.40)
            else:
                trade_usdt = balance_based_cap
            if configured_trade_usdt > 0:
                trade_usdt = min(trade_usdt, configured_trade_usdt)

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
                self.exchange.set_leverage(int(effective_leverage), self.symbol)
            except Exception as e:
                leverage_set = False
                logger.warning(f"Could not set leverage to {int(effective_leverage)}x: {e}")
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
            
            # PURGE OLD ORDERS: Before we place the new one, wipe any "orphaned" orders 
            # left over from previous ticks or duplicate signals.
            try:
                self.exchange.cancel_all_orders(self.symbol)
            except Exception as e:
                logger.debug(f"Order Purge Note: {e}")

            try:
                use_limit = bool(getattr(self, 'use_limit_orders', False))
                real_entry_price = current_price
                
                if use_limit:
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
                    self.pending_entry = {
                        'order_id': str(order_resp.get('id') or (order_resp.get('info', {}) or {}).get('orderId') or ''),
                        'action': action,
                        'trade_id': trade_id,
                        'price': float(price_str),
                        'amount': float(amount_str),
                        'ts': now,
                        'sl': _safe_initial_sl_price(
                            'LONG' if action == "BUY" else "SHORT",
                            float(price_str),
                            signal.get('sl'),
                            getattr(self, 'default_sl_pct', 0.0030),
                        ),
                        'tp': signal.get('tp'),
                        'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                        'score': float(signal.get("score", 0.0) or 0.0),
                        'confidence': float(signal.get("confidence", 0.0) or 0.0),
                        'reason': str(signal.get("reason", "") or ""),
                        'structure_support': signal.get('structure_support'),
                        'structure_resistance': signal.get('structure_resistance'),
                    }
                    self.last_status = f"Maker entry placed @ {price_str}"
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
            
            # The bot uses its own internal dynamic trailing stops and soft limits.
            # We will NOT place hard stop/limit orders on the exchange to avoid
            # market maker hunting and scam wicks.
            pos_side = 'LONG' if action == "BUY" else "SHORT"
            sl_price = _safe_initial_sl_price(
                pos_side,
                real_entry_price,
                signal.get('sl'),
                getattr(self, 'default_sl_pct', 0.0030),
            )
            logger.info(f"Using internal dynamic stop logic. Initial soft SL set at {sl_price:.4f}")
            try:
                tp_price = float(signal.get('tp')) if signal.get('tp') else None
            except (TypeError, ValueError):
                tp_price = None
            
            sl_dist = abs(real_entry_price - sl_price) / real_entry_price if real_entry_price and sl_price else float(getattr(self, 'default_sl_pct', 0.0030))
            filled_pos = {
                'trade_id': trade_id,
                'side': pos_side,
                'entry': real_entry_price,
                'amount': amount,
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
                'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
                'trail_armed': False,
                'sl': sl_price,
                'tp_price': tp_price,
                'structure_support': signal.get('structure_support'),
                'structure_resistance': signal.get('structure_resistance'),
                'native_trailing_activation_pct': float(getattr(self, 'break_even_trigger_pct', 0.0020)),
            }
            self.active_positions.append(filled_pos)
            
            # NATIVE BINANCE TRAILING STOP: Place the official order if enabled
            if getattr(self, 'use_native_trailing_stop', False):
                self._ensure_native_trailing_stop(filled_pos)
            self.trade_count += 1
            self._last_trade_ts = now
            self._log_trade(
                trade_id,
                "ENTRY",
                action,
                current_price,
                amount,
                score=float(signal.get("score", 0.0) or 0.0),
                confidence=float(signal.get("confidence", 0.0) or 0.0),
                reason=str(signal.get("reason", "") or ""),
            )
        except Exception as e:
            logger.error(f"Placement Error: {e}")
            self.last_status = f"Placement error: {e}"

    def close_all_positions(self, symbol: str):
        """Liquidate everything on shutdown with Deep Trace and Global Wipe."""
        # Ensure we have the right symbol formats
        raw_symbol = symbol or self.symbol
        target_symbol = _normalize_futures_symbol(raw_symbol)
        target_id = _market_id_from_symbol(target_symbol).upper()
        
        print(f"\n[SHUTDOWN] Starting Deep Trace Cleanup...")
        print(f"  - Target Symbol: {target_symbol}")
        print(f"  - Target ID: {target_id}")

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
                                self.exchange.create_market_order(target_symbol, side, abs(amt))
                                print(f"    + Liquidation order sent.")
                                try:
                                    self._cleanup_trade_orders(target_symbol)
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

        self.active_positions = []
        self.pending_entry = None
        self.pending_exit = None
        print("[SHUTDOWN] CLEANUP COMPLETE. Account is FLAT.")


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
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self.default_sl_pct = 0.0030
        self.exit_on_reversal_only_in_profit = True
        self._last_trade_ts = 0.0
        self.trade_log_file = "trade_log_spot.csv"
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
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
            net_pnl = pnl - fees
            exit_type = _realized_exit_type(reason, net_pnl)
            self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
            self._log_trade(pos.get("trade_id", 0), "EXIT", "SELL", fill_price, amount, net_pnl, fees, t_type=exit_type)
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
        pos["highest_profit_pct"] = max(float(pos.get("highest_profit_pct", 0.0) or 0.0), best_profit_pct * 100.0)
        pos["lowest_profit_pct"] = min(float(pos.get("lowest_profit_pct", 0.0) or 0.0), profit_pct * 100.0)

        break_even_trigger = float(pos.get("break_even_trigger_pct", self.break_even_trigger_pct) or self.break_even_trigger_pct)
        break_even_buffer = float(pos.get("break_even_buffer_pct", self.break_even_buffer_pct) or self.break_even_buffer_pct)
        if not bool(pos.get("trail_armed", False)) and profit_pct >= break_even_trigger and profit_pct > 0:
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
                        'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                        'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
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
                        net_pnl = pnl - fees
                        exit_type = _realized_exit_type("TAKE_PROFIT", net_pnl)
                        self._record_closed_trade(exit_type, entry, fill_price, pnl, profit_pct * 100, fees)
                        self._log_trade(pos.get("trade_id", 0), "EXIT", "SELL", fill_price, amount, net_pnl, fees, t_type=exit_type)
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
            'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
            'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
            'trail_armed': False,
            'sl': sl_price,
            'tp_price': tp_price,
            'tp_order_id': "",
            'structure_support': signal.get('structure_support'),
            'structure_resistance': signal.get('structure_resistance'),
        })
        self.trade_count += 1
        self._last_trade_ts = now
        self._log_trade(trade_id, "ENTRY", "BUY", fill_price, amount, score=float(signal.get("score", 0.0) or 0.0), confidence=float(signal.get("confidence", 0.0) or 0.0), reason=str(signal.get("reason", "") or ""))
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
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
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

    def _record_closed_trade(self, t_type: str, entry: float, exit_price: float, pnl: float, pnl_pct: float, fees: float):
        net_pnl = float(pnl) - float(fees)
        self.closed_trades.append({
            "type": t_type,
            "entry": float(entry),
            "exit": float(exit_price),
            "pnl": float(net_pnl),
            "pnl_pct": float(pnl_pct),
            "fees": float(fees),
        })
        self.stats_trades += 1
        if net_pnl > 0:
            self.stats_wins += 1
        else:
            self.stats_losses += 1
        self.stats_gross += float(pnl)
        self.stats_fees += float(fees)

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
            current_atr = atr_pct if atr_pct is not None else getattr(self, '_current_atr_pct', 0.02)
            atr_reference = getattr(self, 'atr_reference_pct', 0.02)
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
                    
                    # PRIMARY EXIT: Parabolic SAR flip (SAR moves ABOVE price = trend reversal)
                    if not closed and psar is not None and hold_time > 10:
                        if psar > current_price and profit_pct >= min_net_profit:
                            # SAR flipped bearish while we are in profit — clean exit
                            closed = True
                            exit_type = "PSAR_EXIT"
                        elif psar > current_price and profit_pct < 0:
                            # SAR flipped bearish while in loss — still let SL handle it
                            # but tighten SL to minimize damage
                            tight_sl = current_price * (1 - 0.0008)
                            if tight_sl > float(pos['sl']):
                                pos['sl'] = tight_sl

                    # TRAILING STOP via PSAR: when in profit, trail SL to PSAR
                    if not closed and psar is not None and psar < current_price and profit_pct > 0:
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

                    # PRIMARY EXIT: Parabolic SAR flip (SAR moves BELOW price = trend reversal for short)
                    if not closed and psar is not None and hold_time > 10:
                        if psar < current_price and profit_pct >= min_net_profit:
                            # SAR flipped bullish while we are in profit — clean exit
                            closed = True
                            exit_type = "PSAR_EXIT"
                        elif psar < current_price and profit_pct < 0:
                            # SAR flipped bullish while in loss — tighten SL
                            tight_sl = current_price * (1 + 0.0008)
                            if tight_sl < float(pos['sl']):
                                pos['sl'] = tight_sl

                    # TRAILING STOP via PSAR: when in profit, trail SL to PSAR
                    if not closed and psar is not None and psar > current_price and profit_pct > 0:
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
                    
                    # STRUCTURAL SL CHECK: price hit stop loss
                    if not closed and current_price >= pos['sl']:
                        closed = True
                        profit_pct = (entry - current_price) / entry

                if closed:
                    profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
                    order_side = 'SELL' if side == 'LONG' else 'BUY'

                    pnl = (current_price - entry) * amount if side == 'LONG' else (entry - current_price) * amount
                    fees = (amount * entry * self.fee_rate) + (amount * current_price * self.fee_rate)
                    net_pnl = pnl - fees
                    exit_type = _realized_exit_type(exit_type or "TRAIL_WIN", net_pnl)
                    self.cash_usdt += (pnl - fees)

                    self._record_closed_trade(exit_type, entry, current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(trade_id, "EXIT", order_side, current_price, amount, pnl, fees, t_type=exit_type)
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
                    net_pnl = pnl - fees
                    self.cash_usdt += (pnl - fees)
                    exit_type = "REVERSAL_BANK" if getattr(self, 'exit_on_reversal_only_in_profit', True) else "REVERSAL"
                    self._record_closed_trade(exit_type, current_pos['entry'], current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, fees, t_type=exit_type)
                    self.active_positions = []
                    self._last_trade_ts = now
                    self.last_status = f"{exit_type}: {current_pos['side']} @ ${current_price:.5f} Net ${net_pnl:+,.2f}"
                    if getattr(self, 'exit_on_reversal_only_in_profit', True):
                        return

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
                'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
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
            self._log_trade(
                trade_id,
                "ENTRY",
                action,
                current_price,
                amount,
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
            self.cash_usdt += (pnl - fees)

            profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
            self._record_closed_trade("MANUAL_CLOSE", entry, current_price, pnl, profit_pct * 100, fees)
            self._log_trade(trade_id, "EXIT", order_side, current_price, amount, pnl, fees, t_type="MANUAL_CLOSE")
            self.trade_count += 1

        self.active_positions = []
        self.last_status = "PAPER CLOSE ALL"
