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


def _market_id_from_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace(":USDT", "").replace(":USDC", "")


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


def _compute_trailing_stop(pos: dict, current_price: float) -> float:
    entry = float(pos.get("entry", current_price) or current_price)
    side = str(pos.get("side", "LONG"))
    base_dist = float(pos.get("sl_pct_dist", 0.005) or 0.005)
    break_even_trigger = float(pos.get("break_even_trigger_pct", 0.0015) or 0.0015)
    break_even_buffer = float(pos.get("break_even_buffer_pct", 0.0002) or 0.0002)
    trail_tighten_1 = float(pos.get("trail_tighten_1_pct", 0.0030) or 0.0030)
    trail_tighten_2 = float(pos.get("trail_tighten_2_pct", 0.0060) or 0.0060)
    fee_rate = float(pos.get("fee_rate", 0.0004) or 0.0004)
    min_profit_after_fees = float(pos.get("min_profit_after_fees", 0.001) or 0.001)  # 0.1% min profit after fees

    if side == "LONG":
        best_price = float(pos.get("highest_price", entry) or entry)
        profit_pct = (best_price - entry) / entry if entry else 0.0
        if profit_pct >= trail_tighten_2:
            trail_dist = base_dist * 0.25
        elif profit_pct >= trail_tighten_1:
            trail_dist = base_dist * 0.45
        elif profit_pct >= break_even_trigger:
            trail_dist = base_dist * 0.75
        else:
            trail_dist = base_dist

        trail_sl = best_price * (1 - trail_dist)
        
        # SNIPER PROFIT LOCK: If we have hit break-even trigger, 
        # ensure the SL is at least ABOVE the entry price.
        if profit_pct >= break_even_trigger:
            min_sl = entry * (1 + break_even_buffer)
            trail_sl = max(trail_sl, min_sl)
            
        return trail_sl

    best_price = float(pos.get("lowest_price", entry) or entry)
    profit_pct = (entry - best_price) / entry if entry else 0.0
    if profit_pct >= trail_tighten_2:
        trail_dist = base_dist * 0.25
    elif profit_pct >= trail_tighten_1:
        trail_dist = base_dist * 0.45
    elif profit_pct >= break_even_trigger:
        trail_dist = base_dist * 0.75
    else:
        trail_dist = base_dist

    trail_sl = best_price * (1 + trail_dist)
    
    # SNIPER PROFIT LOCK (SHORT): If we have hit break-even trigger, 
    # ensure the SL is at least BELOW the entry price.
    if profit_pct >= break_even_trigger:
        max_sl = entry * (1 - break_even_buffer)
        trail_sl = min(trail_sl, max_sl)
        
    return trail_sl

class BinanceFuturesExecution:
    def __init__(self, api_key: str, api_secret: str, leverage: int = 5, max_closed_trades: int = 5000, is_demo: bool = True):
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
        self.break_even_trigger_pct = 0.0015
        self.break_even_buffer_pct = 0.0002
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self._last_trade_ts = 0.0
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
        
        self.symbol = "BTC/USDT:USDT"
        self.symbol_id = _market_id_from_symbol(self.symbol)
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
        
        self.max_open_positions = 1
        self.min_balance_floor = 90.0
        self.daily_loss_cap_pct = None
        
        self.leverage_score_weight = 0.3
        self.min_profit_after_fees = 0.0015
        self.exit_on_reversal_only_in_profit = True
        self.use_native_trailing_stop = False
        self.trailing_stop_callback = 0.005
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
            # 1. Standard CCXT
            free = float(balance.get('free', {}).get('USDT', 0.0))
            if free > 0: return free
            
            # 2. Try total
            total = float(balance.get('total', {}).get('USDT', 0.0))
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
                    # Detect which asset to look for based on symbol (USDT or USDC)
                    target_asset = 'USDC' if 'USDC' in str(getattr(self, 'symbol', 'USDT')) else 'USDT'
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
        if not os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, 'w', newline='') as f:
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
            with open(TRADE_LOG_FILE, 'a', newline='') as f:
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
        if not getattr(self, 'use_native_trailing_stop', False):
            return False
        if not isinstance(pos, dict):
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
            # On Futures, total equity is 'total' -> 'USDT' or 'USDC'
            total_equity = float(balance.get('total', {}).get('USDT', 0.0))
            if total_equity == 0:
                total_equity = float(balance.get('total', {}).get('USDC', 0.0))
            
            if total_equity == 0 and 'info' in balance:
                total_equity = float(balance['info'].get('totalWalletBalance', 0.0))
            
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
                    try:
                        self._cleanup_trade_orders(self.symbol, self.active_positions[0] if self.active_positions else None)
                    except Exception as e:
                        logger.debug(f"[SYNC] Open-order cleanup skipped: {e}")
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
                    adopted_pos = {
                        'trade_id': 9999, # Placeholder for adopted trade
                        'side': exch_side,
                        'entry': entry_price,
                        'amount': exch_size,
                        'entry_ts': time.time(), # We don't know exact time, assume now
                        'highest_price': current_price if exch_side == 'LONG' else 0,
                        'lowest_price': current_price if exch_side == 'SHORT' else 0,
                        'highest_profit_pct': 0.0,
                        'sl': entry_price * (0.99 if exch_side == 'LONG' else 1.01), # Default 1% SL if unknown
                        'sl_pct_dist': 0.01,
                        'fee_rate': getattr(self, 'fee_rate', 0.0004),
                        'min_profit_after_fees': getattr(self, 'min_profit_after_fees', 0.0015),
                        'break_even_trigger_pct': float(getattr(self, 'break_even_trigger_pct', 0.0020)),
                        'break_even_buffer_pct': float(getattr(self, 'break_even_buffer_pct', 0.0002)),
                        'native_trailing_activation_pct': float(getattr(self, 'break_even_trigger_pct', 0.0020)),
                    }
                    self.active_positions.append(adopted_pos)
                    self._ensure_native_trailing_stop(adopted_pos)
                else:
                    # SYNC position: Already tracking, just ensure size/side matches
                    local_pos = self.active_positions[0]
                    if abs(exch_size - local_pos['amount']) > 0.000001 or exch_side != local_pos['side']:
                        logger.warning(f"[SYNC] Position mismatch: local {local_pos['side']} {local_pos['amount']}, exchange {exch_side} {exch_size}")
                        local_pos['amount'] = exch_size
                        local_pos['side'] = exch_side
                    self._ensure_native_trailing_stop(local_pos)
        except Exception as e:
            logger.warning(f"Failed to sync positions: {e}")
        
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
                    min_profit = float(pos.get('min_profit_after_fees', 0.001))
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
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                # Trailing TP: ride momentum with super-tight stop
                                tight_sl = current_price * (1 - 0.0010)
                                if tight_sl > float(pos['sl']):
                                    pos['sl'] = tight_sl
                                    pos['sl_pct_dist'] = 0.0010
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"
                                
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct < 0.001:
                        closed = True
                        exit_type = "TTL_EXIT"
                                
                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        # Break-even / Profit protection: move SL to profit floor if possible
                        min_sl_price = entry * (1 + (2.0 * fee_rate) + min_profit)
                        if profit_pct >= float(pos.get('break_even_trigger_pct', 0.0025)):
                            new_sl = max(new_sl, min_sl_price)
                        
                        # Bot-managed trailing stop: update only local SL. The bot records PnL
                        # after its own market close, so exchange-side stops do not create
                        # untracked closes.
                        move_pct = abs(new_sl - float(pos['sl'])) / float(pos['sl']) if float(pos['sl']) > 0 else 1.0
                        if new_sl > float(pos['sl']) and move_pct >= 0.0005:
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[LOCAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f}")
                        
                        # Safety check: if price drops below current SL, close immediately
                        if current_price <= pos['sl']:
                            closed = True
                            profit_pct = (current_price - entry) / entry
                
                else: # SHORT
                    if current_price < float(pos.get('lowest_price', entry) or entry):
                        pos['lowest_price'] = current_price
                    
                    profit_pct = (entry - current_price) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', 0.001))
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
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                # Trailing TP: ride momentum with super-tight stop
                                tight_sl = current_price * (1 + 0.0010)
                                if tight_sl < float(pos['sl']):
                                    pos['sl'] = tight_sl
                                    pos['sl_pct_dist'] = 0.0010
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct < 0.001:
                        closed = True
                        exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        # Break-even / Profit protection: move SL to profit floor if possible
                        min_sl_price = entry * (1 - (2.0 * fee_rate) - min_profit)
                        if profit_pct >= float(pos.get('break_even_trigger_pct', 0.0025)):
                            new_sl = min(new_sl, min_sl_price)
                        
                        # Bot-managed trailing stop: update only local SL.
                        move_pct = abs(new_sl - float(pos['sl'])) / float(pos['sl']) if float(pos['sl']) > 0 else 1.0
                        if new_sl < float(pos['sl']) and move_pct >= 0.0005:
                            old_sl = float(pos['sl'])
                            pos['sl'] = new_sl
                            logger.info(f"[LOCAL_TRAIL] SL moved {old_sl:.4f} -> {new_sl:.4f}")
                        
                        # Safety check: if price drops below current SL, close immediately
                        if current_price >= pos['sl']:
                            closed = True
                            profit_pct = (entry - current_price) / entry

                if closed:
                    profit_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
                    logger.info(f"[FUTURES] Exit {side} @ {current_price:.2f}")

                    order_side = 'SELL' if side == 'LONG' else 'BUY'
                    self._cleanup_trade_orders(self.symbol, pos)

                    if getattr(self, 'use_limit_orders', False):
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
        self.symbol = _normalize_futures_symbol(symbol or self.symbol)
        self.symbol_id = _market_id_from_symbol(self.symbol)
        action = signal['action']
        if action == "HOLD":
            self.last_status = "Signal HOLD"
            return

        try:
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
                        action = str(pending_entry.get("action", action) or action)
                        trade_id = int(pending_entry.get("trade_id", 0) or 0)
                        amount = float(pending_entry.get("amount", 0.0) or 0.0)
                        side = 'BUY' if action == "BUY" else 'SELL'
                        sl_price = float(pending_entry.get("sl", fill_price * (0.993 if action == "BUY" else 1.007)) or fill_price)
                        try:
                            tp_price = float(pending_entry.get("tp")) if pending_entry.get("tp") else None
                        except (TypeError, ValueError):
                            tp_price = None
                        filled_pos = {
                            'trade_id': trade_id,
                            'side': 'LONG' if action == "BUY" else 'SHORT',
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
                    elif age < float(getattr(self, "pending_entry_ttl_seconds", 20) or 20):
                        self.last_status = f"Waiting entry fill @ ${float(pending_entry.get('price', current_price) or current_price):.5f}"
                        return
                    else:
                        if order_id:
                            try:
                                self.exchange.cancel_order(order_id, self.symbol)
                            except Exception:
                                pass
                        self.pending_entry = None
                        self.last_status = "Maker entry expired; retrying"
                        return
                except Exception as e:
                    logger.debug(f"Pending entry check skipped: {e}")

            if self.min_seconds_between_trades and (now - float(self._last_trade_ts)) < float(self.min_seconds_between_trades):
                self.last_status = f"Cooldown ({int(self.min_seconds_between_trades)}s)"
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
            # Determine trade size.
            # If DCA is enabled, use 40% for first entry to leave room for DCA steps (20% each).
            dca_enabled = getattr(self, 'dca_enabled', False)
            if is_dca:
                trade_usdt = balance * 0.20 # 20% per DCA step
            elif dca_enabled:
                trade_usdt = balance * 0.40 # 40% initial
            else:
                trade_usdt = balance * 0.75 # 75% normal
                
            if trade_usdt < 5.0:
                self.last_status = f"Equity too low: ${balance:,.2f}"
                return
            
            # Use dynamic leverage based on signal confidence
            confidence = float(signal.get('confidence', 0.5) or 0.5)
            score = float(signal.get('score', 0.5) or 0.5)
            atr_pct = signal.get('atr_pct')  # Get ATR from signal if available
            current_leverage = self.calculate_dynamic_leverage(confidence, score, atr_pct=atr_pct)
            
            # CRITICAL: Tell Binance to actually use this leverage, otherwise it uses 1x default and fails
            try:
                self.exchange.set_leverage(int(current_leverage), self.symbol)
            except Exception as e:
                logger.warning(f"Could not set leverage to {int(current_leverage)}x: {e}")
                
            amount = (trade_usdt * current_leverage) / current_price
            trade_id = self._next_trade_id
            self._next_trade_id += 1
            
            leverage_info = f" (conf={confidence:.1%}→{current_leverage:.1f}x)" if self.dynamic_leverage_enabled else ""
            logger.info(f"Binance Futures {action}: {amount:.6f} BTC @ {current_price:.2f}{leverage_info}")
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
                        'sl': float(signal.get('sl', current_price * (0.993 if action == "BUY" else 1.007)) or 0.0),
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
            sl_price = signal.get('sl', current_price * (0.993 if action == "BUY" else 1.007))
            logger.info(f"Using internal dynamic stop logic. Initial soft SL set at {sl_price:.4f}")
            try:
                tp_price = float(signal.get('tp')) if signal.get('tp') else None
            except (TypeError, ValueError):
                tp_price = None
            
            sl_dist = abs(current_price - sl_price) / current_price if sl_price else 0.005
            filled_pos = {
                'trade_id': trade_id,
                'side': 'LONG' if action == "BUY" else 'SHORT',
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
        self.break_even_trigger_pct = 0.0015
        self.break_even_buffer_pct = 0.0003
        self.trail_tighten_1_pct = 0.0025
        self.trail_tighten_2_pct = 0.0050
        self._last_trade_ts = 0.0
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
        self.min_balance_floor = 90.0
        self.dynamic_leverage_enabled = False
        self._initial_price_set = True
        self._last_price = None
        self._current_atr_pct = 0.02
        self.min_profit_after_fees = 0.0002
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
        if not os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, 'w', newline='') as f:
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
            with open(TRADE_LOG_FILE, 'a', newline='') as f:
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
                if side == 'LONG':
                    if current_price > float(pos.get('highest_price', entry) or entry):
                        pos['highest_price'] = current_price
                    
                    profit_pct = (current_price - entry) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    # Track worst drawdown for recovery detection
                    pos['lowest_profit_pct'] = min(float(pos.get('lowest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', 0.001))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit
                    
                    # RECOVERY EXIT: if trade dipped badly but bounced back near entry, exit at minimal loss
                    if not closed and float(pos.get('lowest_profit_pct', 0)) < -0.0015 and profit_pct > -0.0005 and hold_time > 30:
                        closed = True
                        exit_type = "RECOVERY_EXIT"
                    
                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                # Instead of closing, activate super-tight trailing stop
                                # Lock SL at (current_price - 0.10% of price) to ride momentum
                                tight_sl = current_price * (1 - 0.0010)  # 0.10% behind current price
                                if tight_sl > float(pos['sl']):
                                    pos['sl'] = tight_sl
                                    pos['sl_pct_dist'] = 0.0010  # Ultra-tight trail from now on
                                # Don't close — let trailing stop handle the exit
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"
                                
                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct < 0.001:
                        closed = True
                        exit_type = "TTL_EXIT"
                                
                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        # Break-even / Profit protection: move SL to profit floor if possible
                        min_sl_price = entry * (1 + (2.0 * fee_rate) + min_profit)
                        if profit_pct >= float(pos.get('break_even_trigger_pct', 0.0025)):
                            new_sl = max(new_sl, min_sl_price)
                        
                        if new_sl > float(pos['sl']):
                            pos['sl'] = new_sl
                        
                        # Safety check: if price drops below current SL, close immediately
                        if current_price <= pos['sl']:
                            closed = True
                            profit_pct = (current_price - entry) / entry
                
                else: # SHORT
                    if current_price < float(pos.get('lowest_price', entry) or entry):
                        pos['lowest_price'] = current_price
                    
                    profit_pct = (entry - current_price) / entry
                    pos['highest_profit_pct'] = max(float(pos.get('highest_profit_pct', 0)), profit_pct)
                    pos['lowest_profit_pct'] = min(float(pos.get('lowest_profit_pct', 0)), profit_pct)
                    fee_rate = float(pos.get('fee_rate', 0.0004))
                    min_profit = float(pos.get('min_profit_after_fees', 0.001))
                    hold_time = time.time() - float(pos.get('entry_ts', time.time()))
                    min_net_profit = (2.0 * fee_rate) + min_profit

                    # RECOVERY EXIT: if trade dipped badly but bounced back near entry, exit at minimal loss
                    if not closed and float(pos.get('lowest_profit_pct', 0)) < -0.0015 and profit_pct > -0.0005 and hold_time > 30:
                        closed = True
                        exit_type = "RECOVERY_EXIT"

                    if not closed:
                        scalp_cfg = getattr(self, 'scalp_config', {})
                        if scalp_cfg:
                            tp_pct = float(scalp_cfg.get('tp_pct', 0.003))
                            if profit_pct >= tp_pct and hold_time >= int(scalp_cfg.get('min_hold_seconds', 10)) and profit_pct >= min_net_profit:
                                # Instead of closing, activate super-tight trailing stop
                                tight_sl = current_price * (1 + 0.0010)  # 0.10% above current price
                                if tight_sl < float(pos['sl']):
                                    pos['sl'] = tight_sl
                                    pos['sl_pct_dist'] = 0.0010
                            elif float(pos.get('highest_profit_pct', 0)) >= float(scalp_cfg.get('fade_trigger_pct', 0.005)) and profit_pct < float(scalp_cfg.get('fade_exit_pct', 0.002)) and hold_time >= 15 and profit_pct >= min_net_profit:
                                closed = True
                                exit_type = "SCALP_EXIT"

                    ttl_seconds = int(getattr(self, 'ttl_exit_seconds', 0))
                    if not closed and ttl_seconds > 0 and hold_time > ttl_seconds and profit_pct < 0.001:
                        closed = True
                        exit_type = "TTL_EXIT"

                    if not closed:
                        new_sl = _compute_trailing_stop(pos, current_price)
                        # Break-even / Profit protection: move SL to profit floor if possible
                        min_sl_price = entry * (1 - (2.0 * fee_rate) - min_profit)
                        if profit_pct >= float(pos.get('break_even_trigger_pct', 0.0025)):
                            new_sl = min(new_sl, min_sl_price)
                        
                        if new_sl < float(pos['sl']):
                            pos['sl'] = new_sl
                        
                        # Safety check: if price drops below current SL, close immediately
                        if current_price >= pos['sl']:
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
                    # REVERSAL: Always allow flipping direction for scalping speed
                    profit_pct = (current_price - current_pos['entry']) / current_pos['entry'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) / current_pos['entry']

                    # Close current position (accept the loss if any)
                    logger.info(f"[REVERSAL] Flipping {current_pos['side']} to {action} (P&L: {profit_pct:+.2%})")
                    order_side = 'SELL' if current_pos['side'] == 'LONG' else 'BUY'
                    pnl = (current_price - current_pos['entry']) * current_pos['amount'] if current_pos['side'] == 'LONG' else (current_pos['entry'] - current_price) * current_pos['amount']
                    fees = (current_pos['amount'] * current_pos['entry'] * self.fee_rate) + (current_pos['amount'] * current_price * self.fee_rate)
                    self.cash_usdt += (pnl - fees)
                    exit_type = "REVERSAL_WIN" if pnl > 0 else "REVERSAL_CUT"
                    self._record_closed_trade(exit_type, current_pos['entry'], current_price, pnl, profit_pct * 100, fees)
                    self._log_trade(current_pos.get("trade_id", 0), "EXIT", order_side, current_price, current_pos['amount'], pnl, fees, t_type=exit_type)
                    self.active_positions = []
                    self._last_trade_ts = now
                    self.last_status = f"{exit_type}: {current_pos['side']} @ ${current_price:.2f} P&L ${pnl:+,.2f}"

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

            # Determine trade size dynamically based on 90% of available balance (leaves 10% buffer for margin/fees)
            trade_usdt = balance * 0.90
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
            
            sl_dist = abs(current_price - signal['sl']) / current_price if signal.get('sl') else 0.005
            use_limit = getattr(self, 'use_limit_orders', False)
            simulated_entry_price = float(signal.get('entry', current_price)) if use_limit else current_price
            
            self.active_positions.append({
                'trade_id': trade_id,
                'side': 'LONG' if action == "BUY" else 'SHORT',
                'entry': simulated_entry_price,
                'amount': amount,
                'entry_ts': now,
                'hold_until_ts': float(signal.get("hold_until_ts", 0.0) or 0.0),
                'highest_price': current_price if action == "BUY" else 0,
                'lowest_price': current_price if action == "SELL" else 0,
                'highest_profit_pct': 0.0,
                'sl_pct_dist': sl_dist,
                'fee_rate': float(self.fee_rate),
                'min_profit_after_fees': 0.001,  # 0.1% minimum profit after fees
                'break_even_trigger_pct': float(self.break_even_trigger_pct),
                'break_even_buffer_pct': float(self.break_even_buffer_pct),
                'trail_tighten_1_pct': float(self.trail_tighten_1_pct),
                'trail_tighten_2_pct': float(self.trail_tighten_2_pct),
                'sl': signal.get('sl', current_price * (0.995 if action == "BUY" else 1.005))
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
