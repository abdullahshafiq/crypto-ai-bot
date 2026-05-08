from __future__ import annotations

import ccxt
import logging
import time
from collections import deque

from ..base import _market_id_from_symbol
from ._balance import BalanceMixin
from ._entry import EntryMixin
from ._exit import ExitMixin
from ._orders import OrderMixin
from ._position import PositionManagerMixin
from ._portfolio import PortfolioMixin
from ._protection import ProtectionMixin
from ._trade_log import TradeLogMixin

logger = logging.getLogger(__name__)


class BinanceFuturesExecution(
    TradeLogMixin,
    BalanceMixin,
    OrderMixin,
    ProtectionMixin,
    ExitMixin,
    EntryMixin,
    PositionManagerMixin,
    PortfolioMixin,
):

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
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }
        })
        if self.is_demo:
            testnet_base = 'https://testnet.binancefuture.com'
            for k, url in list(self.exchange.urls.get('api', {}).items()):
                if not isinstance(url, str) or not k.startswith('fapi'):
                    continue
                self.exchange.urls['api'][k] = url.replace('https://fapi.binance.com', testnet_base)

            logger.info("Binance Futures: Running in TRADITIONAL TESTNET mode")
        else:
            logger.info("Binance Futures: Running in LIVE mode")

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

        self._current_atr_pct = 0.02

        try:
            self.exchange.fapiPrivatePostLeverage({
                'symbol': self.symbol_id,
                'leverage': self.leverage
            })
            logger.info(f"Binance Futures: Leverage set to {self.leverage}x")
        except Exception as e:
            logger.warning(f"Leverage setup note: {e}")
            self.last_status = f"Leverage setup failed: {e}"

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

        leverage = self.leverage_min
        for threshold in sorted(self.leverage_confidence_levels.keys()):
            if confidence >= threshold:
                leverage = self.leverage_confidence_levels[threshold]

        if self.leverage_use_score:
            score_factor = 1.0 + (score - 0.5) * self.leverage_score_weight
            leverage = leverage * score_factor

        atr_volatility_scaling = getattr(self, 'atr_volatility_scaling', False)
        if atr_volatility_scaling:
            current_atr = atr_pct if atr_pct is not None else getattr(self, '_current_atr_pct', 0.5)
            atr_reference = getattr(self, 'atr_reference_pct', 0.5)
            atr_min_multiplier = getattr(self, 'atr_min_multiplier', 0.3)

            if current_atr > 0:
                vol_multiplier = atr_reference / current_atr
                vol_multiplier = max(atr_min_multiplier, min(1.5, vol_multiplier))
                leverage = leverage * vol_multiplier

        risk_multiplier = float(getattr(self, 'learning_risk_multiplier', 1.0) or 1.0)
        leverage = leverage * max(0.5, min(1.0, risk_multiplier))

        leverage = max(self.leverage_min, min(self.leverage_max, leverage))
        return leverage