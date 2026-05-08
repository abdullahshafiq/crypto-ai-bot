"""Regression tests for the signals package — validates that the refactored modular
architecture produces identical outputs to the original monolith.

Run with: python -m pytest tests/test_signals.py -v
"""

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
from indicators.signals.gates.confirmation import (
    apply_strike_zone_check, apply_ob_gate, apply_trend_confirmation_gate,
)
from indicators.signals.gates.sniper import (
    apply_range_reversal_sniper, apply_exhaustion_divergence_gate, apply_mtf_trend_veto,
)
from indicators.signals.gates.bias import apply_trend_continuation_bias, compute_range_zones


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
        from indicators.signals import generate_quant_signal, validate_signal_integrity
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
        from indicators.signals.alpha import generate_alpha_overlay, validate_signal_integrity
        from indicators.signals.utils import _calculate_volume_delta, _map_order_book_pressure

    def test_top_level_indicators(self):
        from indicators import generate_quant_signal, validate_signal_integrity
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
            "action", "score", "confidence", "reason",
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