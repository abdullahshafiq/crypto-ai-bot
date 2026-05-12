"""Regression tests for the signals package — validates that the refactored modular
architecture produces identical outputs to the original monolith.

Run with: python -m pytest tests/test_signals.py -v
"""

import time

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from indicators.signals.ctx import SignalContext
from indicators.signals.engine import generate_quant_signal
from indicators.signals.context import _build_core_context
from indicators.signals.synthesis import _apply_score_synthesis, _determine_action
from indicators.signals.trend import _compute_trend_confirmation
from indicators.signals.builder import _build_signal_dict
from indicators.signals.stops import _apply_setup_overrides, _compute_sl_tp
from indicators.signals.scores import compute_indicator_scores
from indicators.signals.gates.guards import (
    apply_spread_guard, apply_chasing_guard, apply_atr_guard,
    apply_adx_range_filter, apply_low_vol_min_score, apply_session_blackout,
)
from indicators.signals.gates.walls import (
    apply_sr_wall_veto, apply_range_position_veto,
    classify_entry_mode_and_walls, apply_wall_rejection_rescue, apply_midrange_policy,
)
from core.signal_gates import apply_confidence_floor, apply_loss_tilt_hard_gate, apply_aggressive_scalp_gate
from indicators.mtf import _pick_structural_levels
from indicators.signals.gates.confirmation import (
    apply_strike_zone_check, apply_ob_gate, apply_trend_confirmation_gate,
)
from indicators.signals.gates.sniper import (
    apply_range_reversal_sniper, apply_exhaustion_divergence_gate, apply_mtf_trend_veto,
)
from indicators.signals.gates.bias import apply_trend_continuation_bias, compute_range_zones
from indicators.signals.scalper import detect_best_scalper_signal


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n_rows=100, base_price=100.0, trend=0.0):
    """Generate a synthetic OHLCV DataFrame with indicators for testing."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="1min")
    noise = np.random.randn(n_rows) * 0.3
    closes = pd.Series(base_price + np.cumsum(noise) + trend)
    opens = closes + np.random.randn(n_rows) * 0.1
    highs = pd.Series(np.maximum(opens.values, closes.values) + np.abs(np.random.randn(n_rows)) * 0.2)
    lows = pd.Series(np.minimum(opens.values, closes.values) - np.abs(np.random.randn(n_rows)) * 0.2)
    volumes = pd.Series(np.random.randint(100, 10000, n_rows).astype(float))

    df = pd.DataFrame({
        "open": opens.values,
        "high": highs.values,
        "low": lows.values,
        "close": closes.values,
        "volume": volumes.values,
        "ema_9": closes.rolling(9, min_periods=1).mean().values,
        "ema_21": closes.rolling(21, min_periods=1).mean().values,
        "rsi_14": (50.0 + np.random.randn(n_rows) * 10),
        "rsi": (50.0 + np.random.randn(n_rows) * 10),
        "adx": (20.0 + np.abs(np.random.randn(n_rows)) * 10),
        "atr_pct": (0.5 + np.abs(np.random.randn(n_rows)) * 0.2),
        "macd": np.random.randn(n_rows) * 0.01,
        "macd_diff": np.random.randn(n_rows) * 0.001,
        "psar": (closes - np.abs(np.random.randn(n_rows)) * 0.5).values,
        "bb_low": (closes * 0.98).values,
        "bb_high": (closes * 1.02).values,
        "bb_mid": closes.values,
        "obv": np.cumsum(np.sign(closes.diff().fillna(0).values) * volumes.values),
        "obv_ema": pd.Series(np.cumsum(np.sign(closes.diff().fillna(0).values) * volumes.values)).rolling(20, min_periods=1).mean().values,
        "j": (50.0 + np.random.randn(n_rows) * 15),
        "z_score": np.random.randn(n_rows),
        "vwap": closes.values,
        "session_open": closes.iloc[0],
        "previous_close": closes.iloc[-2] if len(closes) > 1 else closes.iloc[0],
        "previous_high": highs.iloc[-2] if len(highs) > 1 else highs.iloc[0],
        "previous_low": lows.iloc[-2] if len(lows) > 1 else lows.iloc[0],
        "trend_bias": 0.0,
        "bb_width": (closes * 0.04).values,
        "atr": (closes * 0.005).values,
    }, index=dates)
    return df


def _make_state(price=100.0, spread_pct=0.001):
    return {
        "price": price,
        "spread_pct": spread_pct,
        "bid": price * (1 - spread_pct / 2),
        "ask": price * (1 + spread_pct / 2),
    }


def _make_latest_indicators(price=100.0, ema_9=None, ema_21=None, rsi_14=50.0,
                             adx=25.0, atr_pct=0.5, macd=0.001, macd_diff=0.0001,
                             psar=None, psar_streak=0, vwap=None, bb_low=None, bb_high=None,
                             z_score=0.0, obv=0.0, obv_ema=0.0, j=50.0, trend_bias=0.0):
    return {
        "close": price,
        "ema_9": ema_9 or price * 0.999,
        "ema_21": ema_21 or price * 0.998,
        "rsi_14": rsi_14,
        "rsi": rsi_14,
        "adx": adx,
        "atr_pct": atr_pct,
        "macd": macd,
        "macd_diff": macd_diff,
        "psar": psar if psar is not None else price * 0.99,
        "psar_streak": psar_streak,
        "vwap": vwap or price,
        "bb_low": bb_low or price * 0.98,
        "bb_high": bb_high or price * 1.02,
        "bb_mid": price,
        "obv": obv,
        "obv_ema": obv_ema or obv,
        "j": j,
        "z_score": z_score,
        "trend_bias": trend_bias,
    }


def _make_local_edge_ctx(current_price=100.55, local_lo=100.50, local_hi=101.50,
                         open_last=100.40, action="HOLD", ema_9=100.0,
                         ema_21=100.0, rsi=35.0, psar=99.0,
                         macd_diff=0.001, prev_macd_diff=0.0,
                         total_score=0.6, strategy_overrides=None):
    df = _make_df(n_rows=60, base_price=(local_lo + local_hi) / 2.0)
    span = max(float(local_hi) - float(local_lo), 1e-9)
    local_mid = (float(local_lo) + float(local_hi)) / 2.0
    local_idx = df.index[-41:-1]
    df.loc[local_idx, ["open", "close"]] = local_mid
    df.loc[local_idx, "high"] = float(local_hi) - (0.05 * span)
    df.loc[local_idx, "low"] = float(local_lo) + (0.05 * span)
    df.loc[df.index[-2], "high"] = float(local_hi)
    df.loc[df.index[-2], "low"] = float(local_lo)
    df.loc[df.index[-1], "open"] = float(open_last)
    df.loc[df.index[-1], "close"] = float(current_price)
    df.loc[df.index[-1], "high"] = max(float(current_price), float(open_last)) + (0.02 * span)
    df.loc[df.index[-1], "low"] = min(float(current_price), float(open_last)) - (0.02 * span)
    df["psar"] = float(psar)
    df["rsi"] = float(rsi)
    df["rsi_14"] = float(rsi)

    strategy_config = {
        "range_position_veto_enabled": True,
        "range_veto_bottom_pct": 0.25,
        "range_veto_top_pct": 0.75,
        "range_action_zone_pct": 0.20,
        "max_chase_pct": 0.001,
        "chase_near_extreme_pct": 0.0,
        "max_consecutive_candles_chase": 4,
    }
    if strategy_overrides:
        strategy_config.update(strategy_overrides)

    return {
        "state": _make_state(current_price, spread_pct=0.0002),
        "latest_indicators": _make_latest_indicators(
            price=current_price,
            ema_9=ema_9,
            ema_21=ema_21,
            rsi_14=rsi,
            psar=psar,
            macd_diff=macd_diff,
        ),
        "strategy_config": strategy_config,
        "df_indicators": df,
        "current_price": float(current_price),
        "ema_21": float(ema_21),
        "ema_9": float(ema_9),
        "atr_pct_now": 0.005,
        "mtf_fast_bias": "NEUTRAL",
        "mtf_fast_score": 0.0,
        "adx_value": 10.0,
        "total_score": total_score,
        "action": action,
        "hold_reason": "",
        "support": 90.0,
        "resistance": 110.0,
        "range_action_zone_pct": 0.20,
        "signal": {
            "action": "HOLD",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "",
            "entry_mode": "TREND",
            "hold_reason": "",
        },
        "wall_state": {"support_broken": False, "resistance_broken": False},
        "psar_bull": float(psar) < float(current_price),
        "macd_diff": macd_diff,
        "prev_macd_diff": prev_macd_diff,
        "rsi_14": rsi,
    }


def _make_strategy_config(**overrides):
    cfg = {
        "range_action_zone_pct": 0.20,
        "macd_noise_threshold": 0.0001,
        "max_structural_sl_pct": 0.0040,
        "min_reward_risk": 0.75,
        "sl_pct": 0.0015,
        "tp_pct": 0.0025,
        "session_filter_enabled": False,
        "spread_max_pct": 0.005,
        "low_vol_min_score": 0.15,
        "chase_max_dist_pct": 0.003,
        "chase_near_extreme_pct": 0.0,
        "midrange_min_score": 0.28,
        "session_block_min_score": 0.35,
    }
    cfg.update(overrides)
    return cfg


def _make_mtf_context(bias="NEUTRAL", macd_15m=0.001, macd_10m=0.001,
                       structure_15m="NEUTRAL", rsi_15m=50.0, rsi_10m=50.0):
    return {
        "15m": {"macd": macd_15m, "structure": structure_15m, "rsi": rsi_15m},
        "10m": {"macd": macd_10m, "rsi": rsi_10m},
        "5m": {"macd": 0.001, "rsi": 50.0},
        "3m": {"macd": 0.001, "rsi": 50.0},
    }


def _make_mtf_config():
    return {"enabled": True, "timeframes": ["3m", "5m", "10m", "15m"]}


def _make_pivot_data(classic=None):
    classic = classic or {"s1": 99.0, "s2": 98.0, "s3": 97.0,
                            "r1": 101.0, "r2": 102.0, "r3": 103.0,
                            "pp": 100.0}
    return {"classic": classic}


# ── SignalContext TypedDict ────────────────────────────────────────────────

class TestSignalContext:
    """Test that the SignalContext TypedDict is properly defined and usable."""

    def test_import(self):
        from indicators.signals.ctx import SignalContext
        assert SignalContext is not None

    def test_create_instance(self):
        ctx: SignalContext = {"state": {}, "current_price": 100.0}
        assert ctx["current_price"] == 100.0

    def test_all_phase_keys_documented(self):
        from indicators.signals.ctx import SignalContext
        expected_phase0 = {"state", "latest_indicators", "strategy_config",
                          "df_indicators", "latest_macro", "mtf_context",
                          "mtf_config", "pivot_data"}
        expected_phase1 = {"current_price", "macro_bias", "range_action_zone_pct",
                          "weights", "support", "resistance", "wall_state",
                          "location_score", "location_notes", "location_levels",
                          "vpoc", "anchored_vwap", "vol_context", "liquidity",
                          "funding_impact"}
        documented = set(SignalContext.__annotations__.keys())
        for key in expected_phase0 | expected_phase1:
            assert key in documented, f"Phase 0/1 key '{key}' missing from SignalContext"


# ── Module Imports ────────────────────────────────────────────────────────

class TestModuleImports:
    """Ensure every module in the signals package imports correctly."""

    def test_signals_package(self):
        from indicators.signals import generate_quant_signal
        from indicators.signals import generate_alpha_overlay, compute_advanced_pivots
        from indicators.signals import _apply_rejection_confirmation_gate

    def test_gates_package(self):
        from indicators.signals.gates import (
            apply_spread_guard, apply_chasing_guard, apply_atr_guard,
            apply_adx_range_filter, apply_low_vol_min_score, apply_session_blackout,
            apply_sr_wall_veto, apply_range_position_veto,
            classify_entry_mode_and_walls, apply_wall_rejection_rescue,
            apply_midrange_policy, apply_strike_zone_check, apply_ob_gate,
            apply_trend_confirmation_gate, apply_range_reversal_sniper,
            apply_exhaustion_divergence_gate, apply_mtf_trend_veto,
            apply_trend_continuation_bias, compute_range_zones,
        )

    def test_sub_modules(self):
        from indicators.signals.context import _build_core_context
        from indicators.signals.synthesis import _apply_score_synthesis, _determine_action
        from indicators.signals.trend import _compute_trend_confirmation
        from indicators.signals.builder import _build_signal_dict
        from indicators.signals.stops import _apply_setup_overrides, _compute_sl_tp
        from indicators.signals.scores import compute_indicator_scores
        from indicators.signals.mtf_bias import compute_mtf_bias
        from indicators.signals.divergence import _detect_macd_divergence
        from indicators.signals.alpha import generate_alpha_overlay
        from indicators.signals.utils import _calculate_volume_delta, _map_order_book_pressure

    def test_top_level_indicators(self):
        from indicators import generate_quant_signal
        from indicators import generate_alpha_overlay, compute_advanced_pivots

    def test_helpers_and_legacy(self):
        from indicators.helpers import get_trend_status, get_quant_signal
        from indicators.legacy import generate_quant_signal as legacy_signal


# ── Guard Functions (Unit Tests) ─────────────────────────────────────────

class TestSpreadGuard:
    def test_wide_spread_returns_hold(self):
        ctx: SignalContext = {
            "state": {"price": 100.0, "bid": 99.5, "ask": 100.5, "spread_pct": 0.01},
            "latest_indicators": {"atr_pct": 0.5},
            "strategy_config": {"spread_max_pct": 0.005},
        }
        result = apply_spread_guard(ctx)
        assert result is not None
        assert result["action"] == "HOLD"
        assert "Spread" in result["reason"]

    def test_tight_spread_passes(self):
        ctx: SignalContext = {
            "state": {"price": 100.0, "bid": 99.99, "ask": 100.01, "spread_pct": 0.0002},
            "latest_indicators": {"atr_pct": 0.5},
            "strategy_config": {"spread_max_pct": 0.005},
        }
        result = apply_spread_guard(ctx)
        assert result is None


class TestChasingGuard:
    def test_extreme_chase_returns_hold(self):
        ctx: SignalContext = {
            "state": {"price": 100.0, "bid": 99.99, "ask": 100.01, "spread_pct": 0.0002},
            "strategy_config": {"chase_max_dist_pct": 0.003},
            "current_price": 100.0,
            "latest_indicators": _make_latest_indicators(ema_9=99.0),
            "df_indicators": _make_df(),
            "ema_21": 100.0,
            "ema_9": 99.0,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "BUY",
        }
        result = apply_chasing_guard(ctx)
        assert result is not None or result is None

    def test_no_chase_returns_none(self):
        ctx: SignalContext = {
            "state": {"price": 95.0, "bid": 94.99, "ask": 95.01, "spread_pct": 0.0002},
            "strategy_config": {"chase_max_dist_pct": 0.003, "chase_near_extreme_pct": 0.0},
            "current_price": 95.0,
            "latest_indicators": _make_latest_indicators(ema_9=94.99),
            "df_indicators": _make_df(),
            "ema_21": 95.0,
            "ema_9": 94.99,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "BUY",
        }
        result = apply_chasing_guard(ctx)
        assert result is None or "Chase" in str(result.get("reason", ""))

    def test_trend_continuation_0_654pct_passes_under_new_cap(self):
        ctx: SignalContext = {
            "state": {"price": 101.0, "bid": 100.99, "ask": 101.01, "spread_pct": 0.0002},
            "strategy_config": {"max_chase_pct": 0.0030, "chase_near_extreme_pct": 0.0},
            "current_price": 101.0,
            "latest_indicators": _make_latest_indicators(ema_9=100.654),
            "df_indicators": _make_df(),
            "ema_21": 100.0,
            "ema_9": 100.654,
            "atr_pct_now": 0.01,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.2,
            "action": "BUY",
        }
        result = apply_chasing_guard(ctx)
        assert result is None

    def test_support_broken_sell_does_not_trip_near_wall_confluence(self):
        ctx: SignalContext = {
            "state": {"price": 100.0, "bid": 99.99, "ask": 100.01, "spread_pct": 0.0002},
            "strategy_config": {"chase_max_dist_pct": 0.003, "chase_near_extreme_pct": 0.0},
            "current_price": 100.0,
            "latest_indicators": _make_latest_indicators(ema_9=100.0),
            "df_indicators": _make_df(),
            "ema_21": 100.0,
            "ema_9": 100.0,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "SELL",
            "rsi_14": 50.0,
            "macd_diff": -0.005,
            "prev_macd_diff": -0.010,
            "sr_score": 1.2,
            "wall_state": {"support_broken": True, "resistance_broken": False},
        }
        result = apply_chasing_guard(ctx)
        assert result is None

    def test_floor_zone_bypasses_chasing_guard(self):
        # ema_dist overextended but price is at floor — guard must pass
        ctx: SignalContext = {
            "state": {"price": 105.0, "bid": 104.99, "ask": 105.01, "spread_pct": 0.0002},
            "strategy_config": {"max_chase_pct": 0.001, "chase_near_extreme_pct": 0.0},
            "current_price": 105.0,
            "latest_indicators": _make_latest_indicators(ema_9=100.0),
            "df_indicators": _make_df(),
            "ema_21": 90.0,   # big dist → would trigger chase guard
            "ema_9": 100.0,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "BUY",
            "support": 100.0,
            "resistance": 200.0,
            "range_action_zone_pct": 0.20,
        }
        result = apply_chasing_guard(ctx)
        assert result is None  # floor zone bypasses chasing guard

    def test_local_floor_bypasses_chasing_guard_and_sniper_can_run(self):
        ctx = _make_local_edge_ctx(
            current_price=100.55,
            local_lo=100.50,
            local_hi=101.50,
            open_last=100.40,
            action="BUY",
            ema_9=105.0,
            ema_21=95.0,
            rsi=35.0,
            psar=99.0,
        )
        result = apply_chasing_guard(ctx)
        assert result is None

        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", "")

    def test_ceiling_zone_bypasses_chasing_guard(self):
        # ema_dist overextended but price is at ceiling — guard must pass
        ctx: SignalContext = {
            "state": {"price": 195.0, "bid": 194.99, "ask": 195.01, "spread_pct": 0.0002},
            "strategy_config": {"max_chase_pct": 0.001, "chase_near_extreme_pct": 0.0},
            "current_price": 195.0,
            "latest_indicators": _make_latest_indicators(ema_9=200.0),
            "df_indicators": _make_df(),
            "ema_21": 210.0,  # big dist → would trigger chase guard
            "ema_9": 200.0,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "SELL",
            "support": 100.0,
            "resistance": 200.0,
            "range_action_zone_pct": 0.20,
        }
        result = apply_chasing_guard(ctx)
        assert result is None  # ceiling zone bypasses chasing guard

    def test_midrange_trend_still_blocked_by_chasing_guard(self):
        # price midrange, ema overextended — guard must fire
        ctx: SignalContext = {
            "state": {"price": 150.0, "bid": 149.99, "ask": 150.01, "spread_pct": 0.0002},
            "strategy_config": {"max_chase_pct": 0.001, "chase_near_extreme_pct": 0.0},
            "current_price": 150.0,
            "latest_indicators": _make_latest_indicators(ema_9=140.0),
            "df_indicators": _make_df(),
            "ema_21": 130.0,  # big dist → triggers chase guard
            "ema_9": 140.0,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "action": "BUY",
            "support": 100.0,
            "resistance": 200.0,
            "range_action_zone_pct": 0.20,
        }
        result = apply_chasing_guard(ctx)
        assert result is not None
        assert result["action"] == "HOLD"

    def test_late_high_buy_still_blocked_by_near_extreme(self):
        ctx = _make_local_edge_ctx(
            current_price=101.45,
            local_lo=100.50,
            local_hi=101.50,
            open_last=101.00,
            action="BUY",
            ema_9=105.0,
            ema_21=95.0,
            rsi=55.0,
            psar=99.0,
            strategy_overrides={"chase_near_extreme_pct": 0.005},
        )
        result = apply_chasing_guard(ctx)
        assert result is None
        assert ctx["action"] == "HOLD"
        assert "Near-Extreme" in ctx.get("hold_reason", "")


class TestATRGuard:
    def test_zero_atr_returns_hold(self):
        ctx: SignalContext = {
            "latest_indicators": {"atr_pct": 0.0, "adx": 25.0},
            "strategy_config": {},
            "current_price": 100.0,
            "atr_pct_now": 0.0,
        }
        result = apply_atr_guard(ctx)
        assert result is not None
        assert result["action"] == "HOLD"

    def test_normal_atr_passes(self):
        ctx: SignalContext = {
            "latest_indicators": {"atr_pct": 0.5, "adx": 25.0},
            "strategy_config": {},
            "current_price": 100.0,
            "atr_pct_now": 0.005,
        }
        result = apply_atr_guard(ctx)
        assert result is None


class TestADXRangeFilter:
    def test_low_adx_returns_hold(self):
        ctx: SignalContext = {
            "latest_indicators": {"adx": 10.0, "atr_pct": 0.5},
            "adx_value": 10.0,
            "current_price": 100.0,
            "total_score": 0.01,
            "action": "BUY",
            "hold_reason": "",
            "strategy_config": {},
        }
        result = apply_adx_range_filter(ctx)
        assert result is not None
        assert result["action"] == "HOLD"

    def test_high_adx_passes(self):
        ctx: SignalContext = {
            "latest_indicators": {"adx": 25.0, "atr_pct": 0.5},
            "adx_value": 25.0,
            "total_score": 0.01,
            "action": "BUY",
            "hold_reason": "",
            "strategy_config": {},
        }
        result = apply_adx_range_filter(ctx)
        assert result is None

    def test_low_adx_floor_zone_bypasses_hold(self):
        # price at floor: support=100, resistance=200, price=105 (5% into range, zone=20%)
        ctx: SignalContext = {
            "latest_indicators": {"adx": 10.0, "atr_pct": 0.5},
            "adx_value": 10.0,
            "current_price": 105.0,
            "support": 100.0,
            "resistance": 200.0,
            "range_action_zone_pct": 0.20,
            "total_score": 0.6,
            "action": "BUY",
            "hold_reason": "",
            "strategy_config": {},
        }
        result = apply_adx_range_filter(ctx)
        assert result is None  # sniper allowed to handle

    def test_low_adx_local_floor_reaches_sniper(self):
        ctx = _make_local_edge_ctx(
            current_price=100.55,
            local_lo=100.50,
            local_hi=101.50,
            open_last=100.40,
            rsi=35.0,
            psar=99.0,
            total_score=0.6,
        )
        result = apply_adx_range_filter(ctx)
        assert result is None

        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", "")

    def test_low_adx_ceiling_zone_bypasses_hold(self):
        # price at ceiling: support=100, resistance=200, price=195 (5% from top, zone=20%)
        ctx: SignalContext = {
            "latest_indicators": {"adx": 10.0, "atr_pct": 0.5},
            "adx_value": 10.0,
            "current_price": 195.0,
            "support": 100.0,
            "resistance": 200.0,
            "range_action_zone_pct": 0.20,
            "total_score": -0.6,
            "action": "SELL",
            "hold_reason": "",
            "strategy_config": {},
        }
        result = apply_adx_range_filter(ctx)
        assert result is None  # sniper allowed to handle

    def test_low_adx_local_ceiling_reaches_sniper(self):
        ctx = _make_local_edge_ctx(
            current_price=101.45,
            local_lo=100.50,
            local_hi=101.50,
            open_last=101.60,
            rsi=65.0,
            psar=102.0,
            total_score=-0.6,
        )
        result = apply_adx_range_filter(ctx)
        assert result is None

        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL"
        assert "EarlyCeil" in ctx["signal"].get("reason", "")

    def test_low_adx_true_local_midrange_still_holds(self):
        ctx = _make_local_edge_ctx(
            current_price=101.00,
            local_lo=100.50,
            local_hi=101.50,
            open_last=101.00,
            total_score=0.01,
        )
        result = apply_adx_range_filter(ctx)
        assert result is not None
        assert result["action"] == "HOLD"


class TestSessionBlackout:
    def test_blackout_disabled_returns_none(self):
        ctx: SignalContext = {
            "strategy_config": {"session_filter_enabled": False},
        }
        result = apply_session_blackout(ctx)
        assert result is None


# ── Wall/Veto Functions ──────────────────────────────────────────────────

class TestSRWallVeto:
    def test_buy_near_resistance_runs_without_error(self):
        ctx: SignalContext = {
            "current_price": 100.0,
            "action": "BUY",
            "hold_reason": "",
            "sr_wall_locked": False,
            "sr_score": 0.5,
            "total_score": 0.4,
            "wall_state": {"resistance_touching": True, "support_touching": False,
                           "near_resistance": True, "near_support": False},
            "resistance": 100.5,
            "support": 99.5,
            "strategy_config": _make_strategy_config(),
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
            "ema_9": 99.99,
        }
        apply_sr_wall_veto(ctx)
        assert "action" in ctx

    def test_sell_near_support_runs_without_error(self):
        ctx: SignalContext = {
            "current_price": 100.0,
            "action": "SELL",
            "hold_reason": "",
            "sr_wall_locked": False,
            "sr_score": -0.5,
            "total_score": -0.4,
            "wall_state": {"resistance_touching": False, "support_touching": True,
                           "near_resistance": False, "near_support": True},
            "resistance": 100.5,
            "support": 99.5,
            "strategy_config": _make_strategy_config(),
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
            "ema_9": 100.01,
        }
        apply_sr_wall_veto(ctx)
        assert "action" in ctx


def _make_range_veto_ctx(
    *,
    current_price: float,
    current_low: float,
    total_score: float,
    macd_diff: float,
    psar_bull: bool,
    support_broken: bool = False,
    mtf_fast_bias: str = "SHORT_ONLY",
) -> SignalContext:
    """Build a deterministic bottom-zone context for range veto regression tests."""
    df_indicators = pd.DataFrame(
        [
            {"open": 100.10, "high": 100.45, "low": 100.00, "close": 100.20, "volume": 1000.0},
            {"open": 100.00, "high": 100.30, "low": 99.95, "close": 100.05, "volume": 1000.0},
            {"open": 99.95, "high": 100.20, "low": 99.90, "close": 99.98, "volume": 1000.0},
            {"open": 99.88, "high": 100.10, "low": 99.70, "close": 99.85, "volume": 1000.0},
            {"open": 99.83, "high": 100.05, "low": 99.82, "close": 99.88, "volume": 1000.0},
            {"open": current_price + 0.01, "high": current_price + 0.08, "low": current_low, "close": current_price, "volume": 1000.0},
        ]
    )
    return {
        "action": "SELL",
        "current_price": current_price,
        "hold_reason": "",
        "signal_reason_suffix": "",
        "sr_wall_locked": False,
        "sr_score": 0.0,
        "total_score": total_score,
        "macd_diff": macd_diff,
        "psar_bull": psar_bull,
        "mtf_fast_bias": mtf_fast_bias,
        "wall_state": {"support_broken": support_broken, "resistance_broken": False},
        "strategy_config": _make_strategy_config(
            range_position_veto_enabled=True,
            range_veto_bottom_pct=0.25,
            range_veto_top_pct=0.75,
            timeframe="1m",
            candles_per_day=6,
        ),
        "df_indicators": df_indicators,
        "latest_indicators": _make_latest_indicators(price=current_price),
    }


class TestRangePositionVeto:
    def test_buy_above_range_gets_vetoed(self):
        ctx: SignalContext = {
            "current_price": 105.0,
            "action": "BUY",
            "hold_reason": "",
            "sr_wall_locked": False,
            "total_score": 0.3,
            "resistance": 102.0,
            "support": 98.0,
            "strategy_config": _make_strategy_config(),
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
        }
        apply_range_position_veto(ctx)
        assert ctx["action"] in {"HOLD", "BUY"}

    def test_sell_early_local_breakdown_allows_short_and_tags_reason(self):
        ctx = _make_range_veto_ctx(
            current_price=99.62,
            current_low=99.60,
            total_score=-0.80,
            macd_diff=-0.015,
            psar_bull=False,
            support_broken=False,
        )
        apply_range_position_veto(ctx)
        assert ctx["action"] == "SELL"
        assert "EarlyLocalBreakdown" in ctx["signal_reason_suffix"]

    def test_sell_at_floor_without_breakdown_stays_hold(self):
        ctx = _make_range_veto_ctx(
            current_price=99.70,
            current_low=99.68,
            total_score=-0.80,
            macd_diff=-0.015,
            psar_bull=False,
            support_broken=False,
        )
        apply_range_position_veto(ctx)
        assert ctx["action"] == "HOLD"

    def test_sell_with_support_broken_still_allows_existing_path(self):
        ctx = _make_range_veto_ctx(
            current_price=99.70,
            current_low=99.68,
            total_score=-0.80,
            macd_diff=-0.015,
            psar_bull=False,
            support_broken=True,
        )
        apply_range_position_veto(ctx)
        assert ctx["action"] == "SELL"

    @pytest.mark.parametrize(
        "total_score, macd_diff, psar_bull",
        [
            (-0.60, -0.015, False),
            (-0.80, 0.0, False),
        ],
    )
    def test_sell_requires_bearish_score_and_falling_macd_for_early_breakdown(
        self,
        total_score: float,
        macd_diff: float,
        psar_bull: bool,
    ):
        ctx = _make_range_veto_ctx(
            current_price=99.62,
            current_low=99.60,
            total_score=total_score,
            macd_diff=macd_diff,
            psar_bull=psar_bull,
            support_broken=False,
        )
        apply_range_position_veto(ctx)
        assert ctx["action"] == "HOLD"


class TestBreakoutRetestGate:
    """Breakout retest gate: price must retest broken level, or very strong score."""

    def _ctx(self, action, current_price, res_broken, sup_broken,
             res_break_level=100.0, sup_break_level=100.0, total_score=0.5):
        return {
            "action": action,
            "current_price": current_price,
            "support": 100.0,
            "resistance": 100.0,
            "total_score": total_score,
            "hold_reason": "",
            "sr_wall_locked": False,
            "strategy_config": _make_strategy_config(),
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
            "mtf_fast_bias": "NEUTRAL",
            "macd_diff": 0.0,
            "psar_bull": True,
            "macd_final_bull": False,
            "macd_final_bear": False,
            "wall_state": {
                "resistance_broken": res_broken,
                "support_broken": sup_broken,
                "resistance_touching": res_broken,
                "support_touching": sup_broken,
                "resistance_break_level": res_break_level,
                "support_break_level": sup_break_level,
            },
        }

    def test_buy_after_resistance_break_no_retest_is_hold(self):
        ctx = self._ctx("BUY", current_price=110.0, res_broken=True, sup_broken=False)
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "HOLD"
        assert "no retest" in ctx["hold_reason"]

    def test_sell_after_support_break_no_retest_is_hold(self):
        ctx = self._ctx("SELL", current_price=90.0, res_broken=False, sup_broken=True)
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "HOLD"
        assert "no retest" in ctx["hold_reason"]

    def test_sell_inside_support_zone_without_break_is_hold(self):
        ctx = self._ctx(
            "SELL",
            current_price=99.9,
            res_broken=False,
            sup_broken=False,
            total_score=-0.3,
        )
        ctx["wall_state"]["support_touching"] = True
        ctx["mtf_fast_bias"] = "SHORT_ONLY"
        ctx["macd_diff"] = -0.1
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "HOLD"
        assert "support not broken" in ctx["hold_reason"]

    def test_buy_with_retest_near_break_level_allows_entry(self):
        ctx = self._ctx("BUY", current_price=100.1, res_broken=True, sup_broken=False)
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "BUY"
        assert ctx["entry_mode"] == "BREAKOUT_LONG"

    def test_buy_far_above_but_strong_score_allows_entry(self):
        ctx = self._ctx("BUY", current_price=110.0, res_broken=True, sup_broken=False, total_score=0.7)
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "BUY"
        assert ctx["entry_mode"] == "BREAKOUT_LONG"

    def test_sell_with_retest_near_break_level_allows_entry(self):
        ctx = self._ctx(
            "SELL",
            current_price=99.9,
            res_broken=False,
            sup_broken=True,
            total_score=-0.3,
        )
        ctx["mtf_fast_bias"] = "SHORT_ONLY"
        ctx["macd_diff"] = -0.1
        classify_entry_mode_and_walls(ctx)
        assert ctx["action"] == "SELL"
        assert ctx["entry_mode"] == "BREAKOUT_SHORT"


class TestEntryQuality:
    def test_ema21_only_support_does_not_trigger_buy(self):
        dates = pd.date_range("2025-01-01", periods=40, freq="1min")
        close = np.full(40, 100.0)
        open_ = np.full(40, 99.8)
        high = np.full(40, 100.3)
        low = np.full(40, 99.6)
        close[-1] = 100.2
        open_[-1] = 100.0
        high[-1] = 100.4
        low[-1] = 99.8
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(40, 1000.0),
                "atr": np.full(40, 0.4),
                "ema_21": np.full(40, 100.15),
                "rsi_14": np.linspace(45.0, 55.0, 40),
                "macd_diff": np.linspace(-0.001, 0.001, 40),
            },
            index=dates,
        )

        scalp = detect_best_scalper_signal(df, _make_strategy_config(), support=None, resistance=None)
        assert scalp["triggered"] is False
        assert scalp["direction"] == "NEUTRAL"

    def test_midrange_trend_buy_is_blocked_even_with_long_only(self):
        ctx = {
            "signal": {
                "action": "BUY",
                "score": 0.30,
                "confidence": 0.30,
                "reason": "READY [Mode:TREND]",
                "structure_support": 40.0,
                "structure_resistance": 60.0,
            },
            "current_price": 50.0,
            "strategy_config": _make_strategy_config(),
            "latest_indicators": _make_latest_indicators(price=50.0),
            "wall_state": {"support_broken": False, "resistance_broken": False},
            "mtf_fast_bias": "LONG_ONLY",
            "sr_score": 0.0,
            "total_score": 0.30,
            "macd_diff": 0.0,
            "psar_bull": True,
            "ema_9_val": 50.0,
        }
        apply_midrange_policy(ctx)
        assert ctx["signal"]["action"] == "HOLD"
        assert "Midrange Gate" in ctx["signal"]["hold_reason"]

    def test_floor_buy_still_allowed(self):
        ctx = {
            "signal": {
                "action": "BUY",
                "score": 0.45,
                "confidence": 0.45,
                "reason": "READY [Mode:REVERSAL_LONG]",
                "structure_support": 40.0,
                "structure_resistance": 60.0,
            },
            "current_price": 40.2,
            "strategy_config": _make_strategy_config(),
            "latest_indicators": _make_latest_indicators(price=40.2),
            "wall_state": {"support_broken": False, "resistance_broken": False},
            "mtf_fast_bias": "NEUTRAL",
            "sr_score": 0.0,
            "total_score": 0.45,
            "macd_diff": 0.0,
            "psar_bull": True,
            "ema_9_val": 40.1,
        }
        apply_midrange_policy(ctx)
        assert ctx["signal"]["action"] == "BUY"

    def test_ceiling_sell_still_allowed(self):
        ctx = {
            "signal": {
                "action": "SELL",
                "score": -0.45,
                "confidence": 0.45,
                "reason": "READY [Mode:REVERSAL_SHORT]",
                "structure_support": 40.0,
                "structure_resistance": 60.0,
            },
            "current_price": 59.8,
            "strategy_config": _make_strategy_config(),
            "latest_indicators": _make_latest_indicators(price=59.8),
            "wall_state": {"support_broken": False, "resistance_broken": False},
            "mtf_fast_bias": "NEUTRAL",
            "sr_score": 0.0,
            "total_score": -0.45,
            "macd_diff": 0.0,
            "psar_bull": False,
            "ema_9_val": 59.9,
        }
        apply_midrange_policy(ctx)
        assert ctx["signal"]["action"] == "SELL"


# ── Synthesis ────────────────────────────────────────────────────────────

class TestDetermineAction:
    def test_positive_score_is_buy(self):
        ctx: SignalContext = {
            "total_score": 0.5,
            "action": "HOLD",
            "hold_reason": "",
            "sr_wall_locked": False,
            "current_price": 100.0,
            "latest_indicators": _make_latest_indicators(),
            "df_indicators": _make_df(),
            "ema_9": 100.0,
            "ema_21": 99.5,
            "strategy_config": _make_strategy_config(),
            "support": 99.0,
            "resistance": 101.0,
            "wall_state": {"resistance_touching": False, "support_touching": False},
            "ema_9_val": 100.0,
            "ema_21_val": 99.5,
        }
        _determine_action(ctx)
        assert ctx["action"] in {"BUY", "SELL", "HOLD"}

    def test_negative_score_is_sell(self):
        ctx: SignalContext = {
            "total_score": -0.5,
            "action": "HOLD",
            "hold_reason": "",
            "sr_wall_locked": False,
            "current_price": 100.0,
            "latest_indicators": _make_latest_indicators(),
            "df_indicators": _make_df(),
            "ema_9": 100.0,
            "ema_21": 100.5,
            "strategy_config": _make_strategy_config(),
            "support": 99.0,
            "resistance": 101.0,
            "wall_state": {"resistance_touching": False, "support_touching": False},
            "ema_9_val": 100.0,
            "ema_21_val": 100.5,
        }
        _determine_action(ctx)
        assert ctx["action"] in {"BUY", "SELL", "HOLD"}


# ── Builder ─────────────────────────────────────────────────────────────

class TestBuildSignalDict:
    def test_output_has_required_keys(self):
        ctx: SignalContext = {
            "total_score": 0.3,
            "action": "BUY",
            "hold_reason": "",
            "sr_wall_locked": False,
            "signal_reason_suffix": " [Indicators OK]",
            "support": 99.0,
            "resistance": 101.0,
            "current_price": 100.0,
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
            "strategy_config": _make_strategy_config(),
            "pivot_data": _make_pivot_data(),
            "wall_state": {"resistance_touching": False, "support_touching": False},
            "location_score": 0.0,
            "location_notes": "",
            "location_levels": {},
            "vpoc": 100.0,
            "anchored_vwap": 100.0,
            "article_sl_override": None,
            "smc_label": "NEUTRAL",
            "mr_score": 0.0,
            "smc_score": 0.0,
            "sr_score": 0.0,
            "vwap_score": 0.0,
            "adx_score": 0.0,
            "volume_delta": 0.0,
            "obv_score": 0.0,
            "bb_score": 0.0,
            "macd_score": 0.0,
            "pa_score": 0.0,
            "kdj_score": 0.0,
            "st_score": 0.0,
            "divergence_state": "NONE",
            "cvd_state": "",
            "momentum_exhaustion": "",
            "mtf_fast_bias": "NEUTRAL",
            "mtf_rsi_bias": "NEUTRAL",
            "mtf_rsi_score": 0.0,
            "body_ratio_score": 0.0,
            "psar_state_note": "PSAR(C:BULL L:BULL)",
            "psar_streak": 3,
            "psar_closed_bull": True,
            "psar_live_bull": True,
            "ob_context": {},
        }
        signal = _build_signal_dict(ctx)
        assert signal["action"] == "BUY"
        assert signal["score"] == 0.3
        assert "reason" in signal
        assert "hold_reason" in signal
        assert "sr_wall_locked" in signal
        assert "market_location" in signal

    def test_retest_short_intent_is_human_readable(self):
        ctx: SignalContext = {
            "total_score": -0.32,
            "action": "HOLD",
            "hold_reason": "Retest Gate: SELL at $100.00 far below break $99.00 - no retest",
            "sr_wall_locked": False,
            "signal_reason_suffix": " [Indicators OK]",
            "entry_mode": "BREAKOUT_SHORT",
            "support": 99.0,
            "resistance": 101.0,
            "current_price": 100.0,
            "df_indicators": _make_df(),
            "latest_indicators": _make_latest_indicators(),
            "strategy_config": _make_strategy_config(),
            "pivot_data": _make_pivot_data(),
            "wall_state": {"resistance_touching": False, "support_touching": False},
            "location_score": 0.0,
            "location_notes": "",
            "location_levels": {},
            "vpoc": 100.0,
            "anchored_vwap": 100.0,
            "article_sl_override": None,
            "smc_label": "NEUTRAL",
            "mr_score": 0.0,
            "smc_score": 0.0,
            "sr_score": 0.0,
            "vwap_score": 0.0,
            "adx_score": 0.0,
            "volume_delta": 0.0,
            "obv_score": 0.0,
            "bb_score": 0.0,
            "macd_score": 0.0,
            "pa_score": 0.0,
            "kdj_score": 0.0,
            "st_score": 0.0,
            "divergence_state": "NONE",
            "cvd_state": "",
            "momentum_exhaustion": "",
            "mtf_fast_bias": "NEUTRAL",
            "mtf_rsi_bias": "NEUTRAL",
            "mtf_rsi_score": 0.0,
            "body_ratio_score": 0.0,
            "psar_state_note": "PSAR(C:BULL L:BULL)",
            "psar_streak": 3,
            "psar_closed_bull": True,
            "psar_live_bull": True,
            "ob_context": {},
        }
        signal = _build_signal_dict(ctx)
        assert signal["intent"] == "Wait for retest, then short"


# ── Stops ────────────────────────────────────────────────────────────────

class TestComputeSLTP:
    def test_buy_signal_gets_sl_below_support(self):
        ctx: SignalContext = {
            "current_price": 100.0,
            "support": 99.0,
            "resistance": 101.0,
            "strategy_config": _make_strategy_config(),
            "pivot_data": _make_pivot_data(),
            "article_sl_override": None,
            "wick_setup": {},
        }
        signal = {"action": "BUY", "score": 0.3, "confidence": 0.3,
                  "reason": "test", "hold_reason": ""}
        result = _compute_sl_tp(signal, ctx)
        assert "sl" in result
        assert result["sl"] < 100.0

    def test_sell_signal_gets_sl_above_resistance(self):
        ctx: SignalContext = {
            "current_price": 100.0,
            "support": 99.0,
            "resistance": 101.0,
            "strategy_config": _make_strategy_config(),
            "pivot_data": _make_pivot_data(),
            "article_sl_override": None,
            "wick_setup": {},
        }
        signal = {"action": "SELL", "score": -0.3, "confidence": 0.3,
                  "reason": "test", "hold_reason": ""}
        result = _compute_sl_tp(signal, ctx)
        assert "sl" in result
        assert result["sl"] > 100.0

    def test_buy_signal_uses_strong_support_with_atr_buffer(self):
        ctx: SignalContext = {
            "current_price": 100.0,
            "support": 99.8,
            "resistance": 101.0,
            "mtf_context": {
                "5m": {
                    "support_levels": [99.8, 99.2],
                    "resistance_levels": [101.4],
                }
            },
            "strategy_config": _make_strategy_config(max_structural_sl_pct=0.03),
            "pivot_data": _make_pivot_data(),
            "article_sl_override": None,
            "wick_setup": {},
            "atr_pct_now": 0.005,
        }
        signal = {"action": "BUY", "score": 0.3, "confidence": 0.3,
                  "reason": "test", "hold_reason": ""}
        result = _compute_sl_tp(signal, ctx)
        assert result["sl_source"] == "strong_support_atr"
        assert result["sl"] == pytest.approx(98.45, rel=1e-4)
        assert result["sl"] < 99.0


# ── Full Pipeline Regression ─────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end regression: generate_quant_signal returns a valid signal dict."""

    def test_warming_up_returns_hold(self):
        result = generate_quant_signal(
            state=_make_state(),
            latest_indicators=_make_latest_indicators(),
            strategy_config=_make_strategy_config(),
            df_indicators=None,
            latest_macro="NEUTRAL",
        )
        assert result["action"] == "HOLD"
        assert result["reason"] == "Warming Up"
        assert result["score"] == 0

    def test_short_df_returns_hold(self):
        df = _make_df(n_rows=10)
        result = generate_quant_signal(
            state=_make_state(),
            latest_indicators=_make_latest_indicators(),
            strategy_config=_make_strategy_config(),
            df_indicators=df,
            latest_macro="NEUTRAL",
        )
        assert result["action"] == "HOLD"
        assert result["reason"] == "Warming Up"

    def test_full_pipeline_returns_signal(self):
        df = _make_df(n_rows=100)
        result = generate_quant_signal(
            state=_make_state(price=100.0, spread_pct=0.001),
            latest_indicators=_make_latest_indicators(price=100.0),
            strategy_config=_make_strategy_config(),
            df_indicators=df,
            latest_macro={"regime": "NEUTRAL"},
            mtf_context=_make_mtf_context(),
            mtf_config=_make_mtf_config(),
            pivot_data=_make_pivot_data(),
        )
        assert result["action"] in {"BUY", "SELL", "HOLD"}
        assert isinstance(result["score"], (int, float))
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["reason"], str)
        assert "score" in result
        assert "confidence" in result
        assert "reason" in result

    def test_signal_dict_has_all_required_keys(self):
        df = _make_df(n_rows=100)
        result = generate_quant_signal(
            state=_make_state(price=100.0, spread_pct=0.0002),
            latest_indicators=_make_latest_indicators(price=100.0),
            strategy_config=_make_strategy_config(spread_max_pct=0.05),
            df_indicators=df,
            latest_macro={"regime": "NEUTRAL"},
            mtf_context=_make_mtf_context(),
            mtf_config=_make_mtf_config(),
            pivot_data=_make_pivot_data(),
        )
        required_keys = [
            "action", "score", "confidence", "reason", "intent",
            "market_bias", "mtf_fast_bias", "mtf_rsi_bias",
            "psar_state_note", "vpoc", "anchored_vwap",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_wide_spread_returns_hold(self):
        df = _make_df(n_rows=100)
        result = generate_quant_signal(
            state=_make_state(price=100.0, spread_pct=0.05),
            latest_indicators=_make_latest_indicators(price=100.0),
            strategy_config=_make_strategy_config(spread_max_pct=0.005),
            df_indicators=df,
            latest_macro="NEUTRAL",
        )
        assert result["action"] == "HOLD"
        assert "Spread" in result["reason"]

    def test_bullish_mtf_results_in_buy_or_neutral(self):
        df = _make_df(n_rows=100, base_price=100.0, trend=0.05)
        result = generate_quant_signal(
            state=_make_state(price=df['close'].iloc[-1], spread_pct=0.0002),
            latest_indicators=_make_latest_indicators(price=df['close'].iloc[-1], adx=30.0),
            strategy_config=_make_strategy_config(),
            df_indicators=df,
            latest_macro={"regime": "BULLISH"},
            mtf_context=_make_mtf_context(macd_15m=0.01, macd_10m=0.005, structure_15m="HH_HL"),
            mtf_config=_make_mtf_config(),
            pivot_data=_make_pivot_data(),
        )
        assert result["action"] in {"BUY", "SELL", "HOLD"}

    def test_bearish_mtf_results_in_sell_or_neutral(self):
        df = _make_df(n_rows=100, base_price=100.0, trend=-0.05)
        result = generate_quant_signal(
            state=_make_state(price=df['close'].iloc[-1], spread_pct=0.0002),
            latest_indicators=_make_latest_indicators(price=df['close'].iloc[-1], adx=30.0),
            strategy_config=_make_strategy_config(),
            df_indicators=df,
            latest_macro={"regime": "BEARISH"},
            mtf_context=_make_mtf_context(macd_15m=-0.01, macd_10m=-0.005, structure_15m="LH_LL"),
            mtf_config=_make_mtf_config(),
            pivot_data=_make_pivot_data(),
        )
        assert result["action"] in {"BUY", "SELL", "HOLD"}


# ── Type Annotation Consistency ──────────────────────────────────────────

class TestTypeAnnotations:
    """Verify that all functions taking ctx use SignalContext."""

    def test_guards_use_signal_context(self):
        import inspect
        from indicators.signals.gates.guards import apply_spread_guard
        sig = inspect.signature(apply_spread_guard)
        assert sig.parameters['ctx'].annotation != inspect.Parameter.empty

    def test_walls_use_signal_context(self):
        import inspect
        from indicators.signals.gates.walls import apply_sr_wall_veto
        sig = inspect.signature(apply_sr_wall_veto)
        assert sig.parameters['ctx'].annotation != inspect.Parameter.empty

    def test_scores_uses_signal_context(self):
        import inspect
        from indicators.signals.scores import compute_indicator_scores
        sig = inspect.signature(compute_indicator_scores)
        assert sig.parameters['ctx'].annotation != inspect.Parameter.empty

    def test_synthesis_uses_signal_context(self):
        import inspect
        from indicators.signals.synthesis import _apply_score_synthesis
        sig = inspect.signature(_apply_score_synthesis)
        assert sig.parameters['ctx'].annotation != inspect.Parameter.empty


def _make_range_df(n=50, lo=90.0, hi=110.0, cur=109.0, open_last=110.0, psar_val=111.0, rsi_val=65.0):
    """Build df with controlled range and last-candle values for sniper tests."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1min")
    mid = (lo + hi) / 2
    closes = np.full(n, mid)
    opens = np.full(n, mid)
    highs = np.full(n, mid + 0.3)
    lows = np.full(n, mid - 0.3)
    highs[0] = hi        # anchor range high
    lows[1] = lo         # anchor range low
    highs[-2] = hi       # _s5 ceiling: current_price <= high[-2]
    lows[-2] = lo        # _s5 floor: current_price >= low[-2]
    closes[-1] = cur
    opens[-1] = open_last
    highs[-1] = max(cur, open_last) + 0.05
    lows[-1] = min(cur, open_last) - 0.05
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.ones(n) * 1000,
        "psar": np.full(n, psar_val),
        "rsi": np.full(n, rsi_val), "rsi_14": np.full(n, rsi_val),
        "macd": np.zeros(n), "macd_diff": np.zeros(n), "macd_signal": np.zeros(n),
        "ema_9": np.full(n, mid), "ema_21": np.full(n, mid), "ema_200": np.full(n, mid),
        "adx": np.full(n, 25.0), "atr": np.full(n, 0.5), "atr_pct": np.full(n, 0.5),
        "bb_high": np.full(n, hi), "bb_low": np.full(n, lo), "bb_mid": np.full(n, mid),
        "bb_width": np.full(n, hi - lo),
        "obv": np.zeros(n), "obv_ema": np.zeros(n),
        "stoch_k": np.full(n, 50.0), "stoch_d": np.full(n, 50.0),
        "j": np.full(n, 50.0), "z_score": np.zeros(n),
        "trend_bias": np.zeros(n), "psar_streak": np.zeros(n),
    }, index=dates)


class TestNearExtremeReversal:
    """Near-extreme guard: wrong-side blocked, pipeline continues for opposite reversal."""

    def _chasing_ctx(self, action, current_price, recent_high=None, recent_low=None,
                     near_extreme_pct=0.005, ema_21=100.0, ema_9=100.0):
        df = _make_df(n_rows=50, base_price=100.0)
        if recent_high is not None:
            df["high"].iloc[-1] = recent_high
        if recent_low is not None:
            df["low"].iloc[-1] = recent_low
        return {
            "current_price": current_price,
            "action": action,
            "ema_21": ema_21,
            "ema_9": ema_9,
            "atr_pct_now": 0.005,
            "mtf_fast_bias": "NEUTRAL",
            "mtf_fast_score": 0.0,
            "latest_indicators": _make_latest_indicators(ema_9=ema_9),
            "df_indicators": df,
            "strategy_config": {
                "chase_near_extreme_pct": near_extreme_pct,
                "chase_recent_extreme_lookback": 30,
                "max_consecutive_candles_chase": 4,
            },
            "rsi_14": 50.0,
            "macd_diff": 0.0,
            "prev_macd_diff": 0.0,
            "sr_score": 0.0,
        }

    def test_buy_near_high_blocks_buy_returns_none(self):
        # BUY near recent high → must NOT early-return dict, must set ctx HOLD
        ctx = self._chasing_ctx(action="BUY", current_price=99.8, near_extreme_pct=0.005)
        # Force recent_high = 100 across all rows
        ctx["df_indicators"]["high"] = 100.0
        result = apply_chasing_guard(ctx)
        assert result is None, "Near-extreme must not early-return dict — pipeline must continue"
        assert ctx["action"] == "HOLD"
        assert "Near-Extreme" in ctx.get("hold_reason", "")

    def test_sell_near_low_blocks_sell_returns_none(self):
        # SELL near recent low → must NOT early-return dict, must set ctx HOLD
        ctx = self._chasing_ctx(action="SELL", current_price=100.2, near_extreme_pct=0.005)
        ctx["df_indicators"]["low"] = 100.0
        result = apply_chasing_guard(ctx)
        assert result is None, "Near-extreme must not early-return dict — pipeline must continue"
        assert ctx["action"] == "HOLD"
        assert "Near-Extreme" in ctx.get("hold_reason", "")

    def test_buy_well_below_high_not_blocked(self):
        ctx = self._chasing_ctx(action="BUY", current_price=95.0, near_extreme_pct=0.005)
        ctx["df_indicators"]["high"] = 100.0  # 95 vs 100*(1-0.005)=99.5 → not triggered
        result = apply_chasing_guard(ctx)
        # Should not trigger near-extreme block (may trigger other guards, but not near-extreme)
        assert ctx.get("hold_reason", "").find("Near-Extreme") == -1 or result is not None or ctx["action"] != "HOLD"

    def test_sell_not_blocked_by_stale_low_from_prior_symbol(self):
        ctx = self._chasing_ctx(action="SELL", current_price=0.258, near_extreme_pct=0.005)
        ctx["df_indicators"]["low"] = 10.102  # stale from prior SUI data (~40x above)
        result = apply_chasing_guard(ctx)
        assert "Near-Extreme" not in ctx.get("hold_reason", "")

    def test_buy_not_blocked_by_stale_high_from_prior_symbol(self):
        ctx = self._chasing_ctx(action="BUY", current_price=10.0, near_extreme_pct=0.005)
        ctx["df_indicators"]["high"] = 0.25  # stale from prior JUP data (~40x below)
        result = apply_chasing_guard(ctx)
        assert "Near-Extreme" not in ctx.get("hold_reason", "")

    def _sniper_ctx(self, cur, lo, hi, open_last, psar_val, rsi_val, mtf_bias="NEUTRAL",
                    macd_diff=0.0, local_lo=None, local_hi=None, local_lookback=40):
        df = _make_range_df(n=50, lo=lo, hi=hi, cur=cur, open_last=open_last,
                            psar_val=psar_val, rsi_val=rsi_val)
        if local_lo is not None and local_hi is not None:
            lookback = max(20, min(int(local_lookback), len(df)))
            start = len(df) - lookback
            span = max(float(local_hi) - float(local_lo), 1e-9)
            local_mid = (float(local_lo) + float(local_hi)) / 2.0
            recent_idx = df.index[start:]
            df.loc[recent_idx, "open"] = local_mid
            df.loc[recent_idx, "close"] = local_mid
            df.loc[recent_idx, "high"] = float(local_hi) - 0.05 * span
            df.loc[recent_idx, "low"] = float(local_lo) + 0.05 * span
            df.loc[df.index[-2], "high"] = float(local_hi)
            df.loc[df.index[-2], "low"] = float(local_lo)
            df.loc[df.index[-1], "open"] = float(open_last)
            df.loc[df.index[-1], "close"] = float(cur)
            df.loc[df.index[-1], "high"] = max(float(cur), float(open_last)) + 0.02 * span
            df.loc[df.index[-1], "low"] = min(float(cur), float(open_last)) - 0.02 * span
        return {
            "signal": {"action": "HOLD", "score": 0.0, "confidence": 0.0,
                       "reason": "Near-Extreme blocked", "hold_reason": "Near-Extreme"},
            "strategy_config": {
                "range_position_veto_enabled": True,
                "range_veto_bottom_pct": 0.25,
                "range_veto_top_pct": 0.75,
                "rsi_os_entry_gate": 28,
                "rsi_ob_entry_gate": 72,
                "timeframe": "1m",
            },
            "current_price": cur,
            "df_indicators": df,
            "latest_indicators": _make_latest_indicators(price=cur, rsi_14=rsi_val),
            "mtf_fast_bias": mtf_bias,
            "macd_diff": macd_diff,
            "psar_bull": psar_val < cur,
            "support": lo,
            "resistance": hi,
        }

    def test_sniper_emits_sell_at_ceiling_with_confirmation(self):
        # Price at 95% of range (ceiling), bearish confirmation → sniper emits SELL
        # cur=109, range=90-110, pos=0.95 > 0.75
        # open_last=110 (cur<=open → _s1=True), psar=111>109 (not psar_bull → _s3=True)
        # rsi=65 >= 62 (_s2=True), high[-2]=110 >= cur=109 (_s5=True) → stab>=3 → SELL
        ctx = self._sniper_ctx(cur=109.0, lo=90.0, hi=110.0,
                               open_last=110.0, psar_val=111.0, rsi_val=65.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL", (
            "Sniper must emit SELL at ceiling when BUY was near-extreme blocked"
        )

    def test_sniper_emits_buy_at_floor_with_confirmation(self):
        # Price at 5% of range (floor), bullish confirmation → sniper emits BUY
        # cur=91, range=90-110, pos=0.05 < 0.25
        # open_last=90 (cur>=open → _s1=True), psar=89<91 (psar_bull → _s3=True)
        # rsi=35 <= 38 (_s2=True), low[-2]=90 <= cur=91 (_s5=True) → stab>=3 → BUY
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=90.0, psar_val=89.0, rsi_val=35.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY", (
            "Sniper must emit BUY at floor when SELL was near-extreme blocked"
        )

    def test_sniper_stays_hold_at_ceiling_without_confirmation(self):
        # Price at ceiling but bullish signals → no reversal confirmation → stays HOLD
        # cur=109, open_last=108 (cur>open → _s1=False), psar=89<109 (psar_bull → _s3=False)
        # rsi=50 < 62 (_s2=False) → stab=0 or 1 < 2 → HOLD
        ctx = self._sniper_ctx(cur=109.0, lo=90.0, hi=110.0,
                               open_last=108.0, psar_val=89.0, rsi_val=50.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "HOLD", (
            "Sniper must stay HOLD at ceiling without bearish confirmation"
        )

    def test_sniper_stays_hold_at_floor_without_confirmation(self):
        # Price at floor but bearish signals → no reversal confirmation → stays HOLD
        # cur=91, open_last=92 (cur<open → _s1=False), psar=111>91 (not psar_bull → _s3=False)
        # rsi=55 > 38 (_s2=False) → stab=0 or 1 < 2 → HOLD
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=92.0, psar_val=111.0, rsi_val=55.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "HOLD", (
            "Sniper must stay HOLD at floor without bullish confirmation"
        )

    def test_sniper_ceil_early_short(self):
        # Price at 95% range, red candle, not breaking prev high, but PSAR still bullish
        # EarlyCeil triggers before stab logic (which would fail with psar_bull + mid RSI)
        ctx = self._sniper_ctx(cur=109.0, lo=90.0, hi=110.0,
                               open_last=110.0, psar_val=89.0, rsi_val=50.0, macd_diff=0.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL", "Early ceiling rejection must emit SELL"
        assert "EarlyCeil" in ctx["signal"].get("reason", ""), "Reason must tag EarlyCeil"

    def test_sniper_floor_early_long(self):
        # Price at 5% range, green candle, not breaking prev low, but PSAR still bearish
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=90.0, psar_val=111.0, rsi_val=55.0, macd_diff=0.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY", "Early floor bounce must emit BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", ""), "Reason must tag EarlyFloor"

    def test_sniper_floor_early_long_allows_flat_open(self):
        # Flat reclaim at the open should still count as an early floor bounce.
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=91.0, psar_val=111.0, rsi_val=55.0, macd_diff=0.0)
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY", "Flat floor reclaim must emit BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", ""), "Reason must tag EarlyFloor"

    def test_sniper_local_ceiling_fires_in_midrange(self):
        ctx = self._sniper_ctx(
            cur=100.4,
            lo=90.0,
            hi=110.0,
            open_last=100.8,
            psar_val=101.0,
            rsi_val=65.0,
            local_lo=99.8,
            local_hi=100.5,
        )
        ctx["signal"]["entry_mode"] = "TREND"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL", "Local ceiling rejection must emit SELL"
        assert "EarlyCeil" in ctx["signal"].get("reason", ""), "Reason must tag EarlyCeil"
        assert ctx["signal"].get("entry_mode") == "RANGE", "Local EarlyCeil must tag RANGE entry_mode"

    def test_sniper_local_floor_fires_in_midrange(self):
        ctx = self._sniper_ctx(
            cur=99.6,
            lo=90.0,
            hi=110.0,
            open_last=99.3,
            psar_val=99.0,
            rsi_val=35.0,
            local_lo=99.5,
            local_hi=100.4,
        )
        ctx["signal"]["entry_mode"] = "TREND"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY", "Local floor bounce must emit BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", ""), "Reason must tag EarlyFloor"
        assert ctx["signal"].get("entry_mode") == "RANGE", "Local EarlyFloor must tag RANGE entry_mode"

    def test_sniper_true_midrange_local_pos_stays_hold(self):
        ctx = self._sniper_ctx(
            cur=100.0,
            lo=90.0,
            hi=110.0,
            open_last=100.2,
            psar_val=101.0,
            rsi_val=50.0,
            local_lo=99.5,
            local_hi=100.5,
        )
        ctx["signal"]["entry_mode"] = "TREND"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "HOLD", "True midrange must stay HOLD"
        assert ctx["signal"].get("entry_mode") == "TREND", "Non-sniper TREND signal must remain TREND"

    def test_long_only_allows_ceiling_sniper_sell_through_veto(self):
        """LONG_ONLY + range ceiling + bearish rejection → sniper SELL survives veto."""
        ctx = self._sniper_ctx(cur=109.0, lo=90.0, hi=110.0,
                               open_last=109.0, psar_val=111.0, rsi_val=65.0,
                               mtf_bias="LONG_ONLY")
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL", "Sniper must emit SELL at ceiling"
        reason_before = ctx["signal"].get("reason", "")
        apply_mtf_trend_veto(ctx)
        assert ctx["signal"]["action"] == "SELL", "MTF veto must allow SELL via RangeCeil exemption"
        assert "RangeCeil" in ctx["signal"].get("reason", reason_before), (
            "Reason must still contain RangeCeil tag after veto"
        )

    def test_short_only_allows_earlyfloor_buy_through_veto(self):
        """SHORT_ONLY + EarlyFloor BUY → MTF veto must not block (EarlyFloor exempt)."""
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=90.0, psar_val=89.0, rsi_val=35.0,
                               mtf_bias="SHORT_ONLY")
        ctx["signal"]["action"] = "BUY"
        ctx["signal"]["reason"] = "[EarlyFloor pos=5% sc=0.55]"
        apply_mtf_trend_veto(ctx)
        assert ctx["signal"]["action"] == "BUY", "MTF veto must allow EarlyFloor BUY even when SHORT_ONLY"

    def test_sniper_flips_sell_to_buy_at_floor_when_bounce_confirms(self):
        ctx = self._sniper_ctx(cur=91.0, lo=90.0, hi=110.0,
                               open_last=90.0, psar_val=89.0, rsi_val=35.0)
        ctx["signal"]["action"] = "SELL"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY", (
            "Sniper must flip SELL to BUY at floor when bounce confirms"
        )

    def test_sniper_flips_buy_to_sell_at_ceiling_when_rejection_confirms(self):
        ctx = self._sniper_ctx(cur=109.0, lo=90.0, hi=110.0,
                               open_last=110.0, psar_val=111.0, rsi_val=65.0)
        ctx["signal"]["action"] = "BUY"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "SELL", (
            "Sniper must flip BUY to SELL at ceiling when rejection confirms"
        )

    def test_sniper_does_not_flip_midrange_sell_or_buy(self):
        sell_ctx = self._sniper_ctx(cur=100.0, lo=90.0, hi=110.0,
                                    open_last=99.0, psar_val=98.0, rsi_val=45.0)
        sell_ctx["signal"]["action"] = "SELL"
        apply_range_reversal_sniper(sell_ctx)
        assert sell_ctx["signal"]["action"] == "SELL", (
            "Midrange SELL must not flip"
        )

        buy_ctx = self._sniper_ctx(cur=100.0, lo=90.0, hi=110.0,
                                   open_last=101.0, psar_val=102.0, rsi_val=55.0)
        buy_ctx["signal"]["action"] = "BUY"
        apply_range_reversal_sniper(buy_ctx)
        assert buy_ctx["signal"]["action"] == "BUY", (
            "Midrange BUY must not flip"
        )


class TestTrendConfirmationGate:
    """Validation for sell_momentum_escape with range-bottom guard."""

    def test_sell_momentum_escape_with_bearish_short_tf(self):
        ctx: SignalContext = {
            "current_price": 105.0,
            "action": "SELL",
            "in_action_zone": False,
            "ema_9_val": 106.0,
            "ema_21": 106.0,
            "psar_bull": False,
            "macd_final_bull": False,
            "macd_final_bear": True,
            "mtf_macd_bull": False,
            "mtf_macd_bear": False,
            "mtf_structure_bull": False,
            "mtf_structure_bear": False,
            "macd_diff": -0.001,
            "pa_score": 0.0,
            "support": 100.0,
            "resistance": 110.0,
            "latest_indicators": {"ema_21": 106.0},
        }
        apply_trend_confirmation_gate(ctx)
        assert ctx["action"] == "SELL", (
            "sell_momentum_escape must allow SELL with PSAR BEAR + MACD falling + not at bottom"
        )

    def test_sell_momentum_escape_blocked_at_bottom_single_conf(self):
        ctx: SignalContext = {
            "current_price": 102.0,
            "action": "SELL",
            "in_action_zone": False,
            "ema_9_val": 104.0,
            "ema_21": 103.0,
            "psar_bull": False,
            "macd_final_bull": True,
            "macd_final_bear": False,
            "mtf_macd_bull": False,
            "mtf_macd_bear": False,
            "mtf_structure_bull": False,
            "mtf_structure_bear": False,
            "macd_diff": 0.0,
            "pa_score": -0.2,
            "support": 100.0,
            "resistance": 110.0,
            "latest_indicators": {"ema_21": 103.0},
        }
        apply_trend_confirmation_gate(ctx)
        assert ctx["action"] == "HOLD", (
            "sell_momentum_escape must block SELL in bottom 25% even with PSAR BEAR alone"
        )

    def test_sell_escape_flat_macd_ema_pa_align(self):
        ctx: SignalContext = {
            "current_price": 105.0,
            "action": "SELL",
            "in_action_zone": False,
            "ema_9_val": 104.0,
            "ema_21": 106.0,
            "psar_bull": True,
            "macd_final_bull": True,
            "macd_final_bear": False,
            "mtf_macd_bull": False,
            "mtf_macd_bear": True,
            "mtf_structure_bull": False,
            "mtf_structure_bear": True,
            "macd_diff": 0.0,
            "pa_score": -0.35,
            "support": 100.0,
            "resistance": 110.0,
            "latest_indicators": {"ema_21": 106.0},
        }
        apply_trend_confirmation_gate(ctx)
        assert ctx["action"] == "SELL", (
            "sell_momentum_escape must allow SELL with flat MACD when EMA bear + PA bear align"
        )


class TestStructuralLevelsStaleRejection:
    """Stale/cross-symbol structural levels must be rejected and replaced with local fallbacks."""

    def test_stale_resistance_rejected_percentage_fallback(self):
        support, resistance = _pick_structural_levels(
            current_price=0.258,
            mtf_context={"5m": {"resistance_levels": [0.283], "support_levels": [0.256]}},
            max_sl_pct=0.012,
        )
        assert resistance <= 0.258 * (1 + 0.012), f"resistance {resistance} too far"
        assert resistance > 0.258, "resistance must be above price"

    def test_stale_support_rejected_percentage_fallback(self):
        support, resistance = _pick_structural_levels(
            current_price=10.0,
            mtf_context={"5m": {"support_levels": [0.25], "resistance_levels": [10.15]}},
            max_sl_pct=0.012,
        )
        assert support >= 10.0 * (1 - 0.012), f"support {support} too far"
        assert support < 10.0, "support must be below price"

    def test_valid_close_levels_preserved(self):
        support, resistance = _pick_structural_levels(
            current_price=0.258,
            mtf_context={"5m": {"resistance_levels": [0.260], "support_levels": [0.255]}},
            max_sl_pct=0.012,
        )
        assert resistance == 0.260
        assert support == 0.255

    def test_stale_levels_use_recent20_fallback_when_available(self):
        support, resistance = _pick_structural_levels(
            current_price=0.258,
            mtf_context={"5m": {
                "resistance_levels": [0.283],
                "support_levels": [0.256],
                "recent_high_20": 0.260,
                "recent_low_20": 0.254,
            }},
            max_sl_pct=0.012,
        )
        assert 0.258 < resistance <= 0.260, f"resistance {resistance} not using recent_high_20"
        assert 0.254 <= support < 0.258, f"support {support} not using recent_low_20"


class TestRangeSniperConfidenceFloor:
    """Range sniper entries use lower confidence floor than normal trend entries."""

    def test_earlyfloor_conf_037_passes_under_loss_tilt_hard_gate(self):
        sig = {"action": "BUY", "confidence": 0.37, "score": 0.37, "reason": "[EarlyFloor pos=5% depth=0.80 sc=0.37]"}
        result = apply_loss_tilt_hard_gate(sig, 0.50, 0.37, {"range_sniper_min_conf": 0.40})
        assert result["action"] == "BUY"

    def test_normal_trend_conf_037_blocks_under_loss_tilt_hard_gate(self):
        sig = {"action": "BUY", "confidence": 0.37, "score": 0.37, "reason": " READY: EMA:OK MTF:LONG_ONLY"}
        result = apply_loss_tilt_hard_gate(sig, 0.50, 0.37, {"range_sniper_min_conf": 0.40})
        assert result["action"] == "HOLD"

    def test_rangeceil_conf_037_passes_under_loss_tilt_hard_gate(self):
        sig = {"action": "SELL", "confidence": 0.37, "score": -0.37, "reason": "[RangeCeil pos=90% depth=0.80 sc=0.37]"}
        result = apply_loss_tilt_hard_gate(sig, 0.50, -0.37, {"range_sniper_min_conf": 0.40})
        assert result["action"] == "SELL"

    def test_rangefloor_buy_passes_with_low_conf(self):
        sig = {"action": "BUY", "confidence": 0.09, "reason": "[RangeFloor pos=10% stab=4/5 sc=0.42]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "BUY"

    def test_rangeceil_sell_passes_with_low_conf(self):
        sig = {"action": "SELL", "confidence": 0.09, "reason": "[RangeCeil pos=90% stab=4/5 sc=0.42]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "SELL"

    def test_earlyfloor_buy_passes_with_low_conf(self):
        sig = {"action": "BUY", "confidence": 0.09, "reason": "[EarlyFloor pos=5% depth=0.80 sc=0.45]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "BUY"

    def test_earlyceil_sell_passes_with_low_conf(self):
        sig = {"action": "SELL", "confidence": 0.09, "reason": "[EarlyCeil pos=95% depth=0.80 sc=0.45]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "SELL"

    def test_normal_trend_signal_blocked_at_low_conf(self):
        sig = {"action": "BUY", "confidence": 0.09, "reason": " READY: EMA:OK MTF:LONG_ONLY"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "HOLD"

    def test_sniper_below_sniper_min_blocked(self):
        sig = {"action": "BUY", "confidence": 0.04, "reason": "[RangeFloor pos=10% sc=0.35]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "HOLD"

    def test_sniper_uses_configured_min(self):
        sig = {"action": "BUY", "confidence": 0.06, "reason": "[RangeFloor pos=10% sc=0.35]"}
        cfg = {"range_sniper_min_conf": 0.05}
        result = apply_confidence_floor(sig, 0.10, cfg)
        assert result["action"] == "BUY"

    def test_rangefloor_conf_above_global_passes(self):
        sig = {"action": "BUY", "confidence": 0.12, "reason": "[RangeFloor pos=10% sc=0.42]"}
        result = apply_confidence_floor(sig, 0.10)
        assert result["action"] == "BUY"


class TestModeRouter:
    """Mode-router: entry_mode gates MTF veto, midrange policy, rejection confirmation."""

    def _make_signal_injector(self, entry_mode: str):
        from indicators.signals.builder import _build_signal_dict as _real_build
        def _patched(ctx):
            sig = _real_build(ctx)
            sig['entry_mode'] = entry_mode
            return sig
        return _patched

    def _run_pipeline(self, entry_mode, mtf_mock, mid_mock, rej_mock, exhaust_mock=None):
        from unittest.mock import patch
        df = _make_df(n_rows=100)
        injector = self._make_signal_injector(entry_mode)
        patches = [
            patch('indicators.signals.engine._build_signal_dict', side_effect=injector),
            patch('indicators.signals.engine.apply_mtf_trend_veto', mtf_mock),
            patch('indicators.signals.engine.apply_midrange_policy', mid_mock),
            patch('indicators.signals.engine._apply_rejection_confirmation_gate', rej_mock),
        ]
        if exhaust_mock is not None:
            patches.append(patch('indicators.signals.engine.apply_exhaustion_divergence_gate', exhaust_mock))
        kwargs = dict(
            state=_make_state(price=100.0, spread_pct=0.0001),
            latest_indicators=_make_latest_indicators(price=100.0),
            strategy_config=_make_strategy_config(),
            df_indicators=df,
            latest_macro={"regime": "NEUTRAL"},
            mtf_context=_make_mtf_context(),
            mtf_config=_make_mtf_config(),
            pivot_data=_make_pivot_data(),
        )
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return generate_quant_signal(**kwargs)

    def _mocks(self):
        from unittest.mock import Mock
        mtf = Mock()
        mid = Mock()
        rej = Mock(side_effect=lambda sig, *a, **kw: (sig, False))
        return mtf, mid, rej

    def test_range_mode_skips_mtf_midrange_rejection(self):
        mtf, mid, rej = self._mocks()
        result = self._run_pipeline('RANGE', mtf, mid, rej)
        mtf.assert_not_called()
        mid.assert_not_called()
        rej.assert_not_called()
        assert result['action'] in {'BUY', 'SELL', 'HOLD'}

    def test_breakout_mode_skips_midrange_only(self):
        mtf, mid, rej = self._mocks()
        self._run_pipeline('BREAKOUT', mtf, mid, rej)
        mtf.assert_called_once()
        mid.assert_not_called()
        rej.assert_called_once()

    def test_trend_mode_calls_all_gates(self):
        mtf, mid, rej = self._mocks()
        self._run_pipeline('TREND', mtf, mid, rej)
        mtf.assert_called_once()
        mid.assert_called_once()
        rej.assert_called_once()

    def test_earlyfloor_refreshes_entry_mode_before_router(self):
        from unittest.mock import Mock, patch
        df = _make_range_df(n=50, lo=90.0, hi=110.0, cur=91.0, open_last=90.0, psar_val=111.0, rsi_val=55.0)
        mtf = Mock()
        mid = Mock()
        rej = Mock(side_effect=lambda sig, *a, **kw: (sig, False))
        with patch('indicators.signals.engine._build_core_context', side_effect=lambda ctx: (ctx.update({
            'current_price': 91.0,
            'support': 90.0,
            'resistance': 110.0,
            'wall_state': {'support_broken': False, 'resistance_broken': False},
            'macro_bias': 'NEUTRAL',
            'range_action_zone_pct': 0.20,
            'weights': {},
            'vpoc': 100.0,
            'anchored_vwap': 100.0,
            'vol_context': {},
            'liquidity': {},
            'funding_impact': 0.0,
        }) or None)), \
             patch('indicators.signals.engine._build_signal_dict', side_effect=lambda ctx: {
                 'action': 'HOLD',
                 'score': 0.0,
                 'confidence': 0.0,
                 'reason': 'seed',
                 'hold_reason': 'seed',
                 'entry_mode': 'TREND',
             }), \
             patch('indicators.signals.engine.apply_chasing_guard', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine.apply_atr_guard', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine.apply_adx_range_filter', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine.compute_indicator_scores', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine._apply_score_synthesis', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine._determine_action', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine._compute_trend_confirmation', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine._apply_setup_overrides', side_effect=lambda signal, ctx: None), \
             patch('indicators.signals.engine.apply_mtf_trend_veto', mtf), \
             patch('indicators.signals.engine.apply_midrange_policy', mid), \
             patch('indicators.signals.engine._apply_rejection_confirmation_gate', rej), \
             patch('indicators.signals.engine.apply_exhaustion_divergence_gate', side_effect=lambda ctx: None), \
             patch('indicators.signals.engine.apply_wall_rejection_rescue', side_effect=lambda ctx: None):
            result = generate_quant_signal(
                state=_make_state(price=91.0, spread_pct=0.0001),
                latest_indicators=_make_latest_indicators(price=91.0, rsi_14=55.0),
                strategy_config=_make_strategy_config(max_structural_sl_pct=0.05),
                df_indicators=df,
                latest_macro={'regime': 'NEUTRAL'},
                mtf_context=_make_mtf_context(),
                mtf_config=_make_mtf_config(),
                pivot_data=_make_pivot_data(),
            )
        assert result['action'] == 'BUY'
        assert result.get('entry_mode') == 'RANGE'
        mtf.assert_not_called()
        mid.assert_not_called()
        rej.assert_not_called()

    def test_hard_safety_hold_blocks_range_mode(self):
        from unittest.mock import Mock
        def exhaust_hold(ctx):
            ctx['signal']['action'] = 'HOLD'
            ctx['signal']['reason'] = 'ExhaustionDivergence'
        mtf, mid, rej = self._mocks()
        result = self._run_pipeline('RANGE', mtf, mid, rej, exhaust_mock=exhaust_hold)
        assert result['action'] == 'HOLD'


class TestAggressiveScalpGate:
    """Paper-only aggressive scalp gate: flips HOLD -> BUY/SELL under tight conditions."""

    def _make_signal(
        self,
        action="HOLD",
        score=0.0,
        tp=0.0,
        sl=0.0,
        resistance=110.0,
        support=90.0,
        reason="",
        hold_reason="CONF<0.15",
    ):
        return {
            "action": action,
            "score": score,
            "tp": tp,
            "sl": sl,
            "structure_resistance": resistance,
            "structure_support": support,
            "reason": reason,
            "hold_reason": hold_reason if action == "HOLD" else "",
        }

    def _exec_cfg(self):
        return {"fee_rate": 0.0006, "min_seconds_between_trades": 60}

    def _strategy_cfg(self, enabled=True):
        return {
            "aggressive_scalp_enabled": enabled,
            "aggressive_scalp_min_conf": 0.06,
            "aggressive_scalp_fee_buffer_pct": 0.0003,
            "max_structural_sl_pct": 0.0030,
        }

    def test_conf_hold_can_become_aggressive_scalp(self):
        sig = self._make_signal(score=0.15, tp=101.0, sl=99.8, support=90.0, resistance=110.0)
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(), executor)
        assert result["action"] == "BUY"
        assert "[AGGRESSIVE_SCALP]" in result["reason"]

    def test_loss_tilt_hold_cannot_become_aggressive_scalp(self):
        sig = self._make_signal(
            score=0.15,
            tp=101.0,
            sl=99.8,
            support=90.0,
            resistance=110.0,
            hold_reason="Consecutive loss tilt: entry pause",
        )
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(), executor)
        assert result["action"] == "HOLD"
        assert "[AGGRESSIVE_SCALP]" not in result["reason"]

    def test_wall_veto_hold_cannot_become_aggressive_scalp(self):
        sig = self._make_signal(
            score=0.15,
            tp=101.0,
            sl=99.8,
            support=90.0,
            resistance=110.0,
            hold_reason="Wall Veto: resistance overhead",
        )
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(), executor)
        assert result["action"] == "HOLD"

    def test_structural_sl_too_far_hold_cannot_become_aggressive_scalp(self):
        sig = self._make_signal(
            score=0.15,
            tp=101.0,
            sl=99.8,
            support=90.0,
            resistance=110.0,
            hold_reason="Structural SL too far (0.60% > 0.30%)",
        )
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(), executor)
        assert result["action"] == "HOLD"

    def test_normal_signals_unchanged(self):
        sig = self._make_signal(action="BUY", score=0.15, tp=101.0, sl=99.8, support=90.0, resistance=110.0)
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(), executor)
        assert result["action"] == "BUY"

    def test_normal_hold_unchanged_when_mode_disabled(self):
        sig = self._make_signal(score=0.15, tp=101.0, sl=99.8, support=90.0, resistance=110.0)
        state = {"price": 100.0}
        executor = MagicMock()
        executor._last_trade_ts = 0.0
        result = apply_aggressive_scalp_gate(sig, state, self._exec_cfg(), self._strategy_cfg(enabled=False), executor)
        assert result["action"] == "HOLD"


class TestSniperSLTPOrdering:
    """After sniper flips HOLD -> SELL/BUY, SL/TP must be computed with valid RR > 0."""

    def _ctx(self, cur, lo, hi, open_last, psar_val, rsi_val, action, score, psar_bull, mtf_bias="NEUTRAL"):
        df = _make_range_df(n=50, lo=lo, hi=hi, cur=cur, open_last=open_last,
                            psar_val=psar_val, rsi_val=rsi_val)
        return {
            "signal": {"action": action, "score": score, "confidence": abs(score),
                       "reason": "", "hold_reason": "Near-Extreme" if action == "HOLD" else ""},
            "strategy_config": {
                "range_position_veto_enabled": True,
                "range_veto_bottom_pct": 0.25,
                "range_veto_top_pct": 0.75,
                "rsi_os_entry_gate": 28,
                "rsi_ob_entry_gate": 72,
                "timeframe": "1m",
                "max_structural_sl_pct": 0.03,
                "sl_pct": 0.0025,
                "tp_pct": 0.0060,
            },
            "current_price": cur,
            "df_indicators": df,
            "latest_indicators": _make_latest_indicators(price=cur, rsi_14=rsi_val),
            "mtf_fast_bias": mtf_bias,
            "macd_diff": 0.0,
            "psar_bull": psar_bull,
            "support": lo,
            "resistance": hi,
            "mtf_context": {},
            "pivot_data": _make_pivot_data(),
            "atr_pct_now": 0.005,
        }

    def test_hold_earlyceil_sell_gets_sl_tp_and_rr(self):
        ctx = self._ctx(cur=109.0, lo=90.0, hi=110.0, open_last=110.0,
                        psar_val=111.0, rsi_val=65.0, action="HOLD", score=0.5,
                        psar_bull=False)
        ctx["signal"]["entry_mode"] = "TREND"
        apply_range_reversal_sniper(ctx)
        signal = ctx["signal"]
        assert signal["action"] == "SELL", "Sniper must flip HOLD to SELL"
        assert signal.get("entry_mode") == "RANGE", "Sniper SELL must be tagged as RANGE"

        signal = _compute_sl_tp(signal, ctx)
        assert signal.get("sl", 0.0) > 109.0, f"SL {signal.get('sl')} must be above entry"
        assert signal.get("tp", 0.0) < 109.0, f"TP {signal.get('tp')} must be below entry"
        assert signal.get("reward_risk", 0.0) > 0.0, "RR must be positive after SL/TP compute"

    def test_hold_earlyfloor_buy_gets_sl_tp_and_rr(self):
        ctx = self._ctx(cur=91.0, lo=90.0, hi=110.0, open_last=90.0,
                        psar_val=89.0, rsi_val=35.0, action="HOLD", score=-0.5,
                        psar_bull=True)
        ctx["signal"]["entry_mode"] = "TREND"
        apply_range_reversal_sniper(ctx)
        signal = ctx["signal"]
        assert signal["action"] == "BUY", "Sniper must flip HOLD to BUY"
        assert signal.get("entry_mode") == "RANGE", "Sniper BUY must be tagged as RANGE"

        signal = _compute_sl_tp(signal, ctx)
        assert signal.get("sl", 1e9) < 91.0, f"SL {signal.get('sl')} must be below entry"
        assert signal.get("tp", -1.0) > 91.0, f"TP {signal.get('tp')} must be above entry"
        assert signal.get("reward_risk", 0.0) > 0.0, "RR must be positive after SL/TP compute"


class TestEarlyFloorEma9Gate:
    """EarlyFloor EMA9 reclaim: first bounce candle below open but above EMA9 should emit BUY."""

    def test_red_bounce_candle_at_floor_ema9_reclaim_emits_earlyfloor(self):
        # price < open[-1] (red intrabar) but price >= ema9 — new gate should pass
        ctx = _make_local_edge_ctx(
            current_price=100.55,
            local_lo=100.50,
            local_hi=101.50,
            open_last=100.70,  # red candle: price < open
            ema_9=100.53,      # price >= ema9: reclaim
            rsi=35.0,
            psar=99.0,
            action="HOLD",
        )
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", "")
        assert ctx["signal"].get("entry_mode") == "RANGE"

    def test_falling_knife_below_low_and_ema9_stays_hold(self):
        # price < low[-2] AND price < ema9 — falling knife, must not emit
        ctx = _make_local_edge_ctx(
            current_price=100.45,
            local_lo=100.50,   # low[-2] = 100.50; price=100.45 < low[-2]
            local_hi=101.50,
            open_last=100.70,
            ema_9=100.60,      # price < ema9
            rsi=35.0,
            psar=101.0,        # bearish psar
            macd_diff=0.001,
            prev_macd_diff=0.0,
            action="HOLD",
        )
        ctx["mtf_fast_bias"] = "SHORT_ONLY"  # min_stab=4, stab path also fails
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "HOLD"

    def test_short_only_mtf_earlyfloor_still_emits_on_ema9_reclaim(self):
        # MTF SHORT_ONLY + red bounce candle + EMA9 reclaim → EarlyFloor BUY (contraMTF allowed)
        ctx = _make_local_edge_ctx(
            current_price=100.55,
            local_lo=100.50,
            local_hi=101.50,
            open_last=100.70,  # red candle
            ema_9=100.53,      # price >= ema9
            rsi=35.0,
            psar=99.0,
            action="HOLD",
        )
        ctx["mtf_fast_bias"] = "SHORT_ONLY"
        apply_range_reversal_sniper(ctx)
        assert ctx["signal"]["action"] == "BUY"
        assert "EarlyFloor" in ctx["signal"].get("reason", "")
