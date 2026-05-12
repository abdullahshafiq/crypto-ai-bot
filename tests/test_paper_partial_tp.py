import time

import pytest

import execution.paper as paper_module
from execution.paper import PaperFuturesExecution


def _make_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_module, "TRADE_LOG_FILE", str(tmp_path / "trade_log_demo_futures.csv"))
    executor = PaperFuturesExecution(starting_balance_usdt=1000.0, leverage=5, fee_rate=0.0006)
    executor.scalp_config = {
        "runner_enabled": True,
        "tp_pct": 0.0060,
        "min_hold_seconds": 20,
        "runner_pullback_pct": 0.0020,
        "runner_min_lock_pct": 0.0040,
        "runner_exchange_tp_multiplier": 1.0,
        "runner_partial_exit_pct": 0.45,
        "fade_trigger_pct": 0.0060,
        "fade_exit_pct": 0.0030,
    }
    executor.break_even_trigger_pct = 0.0060
    executor.profit_trailing_activation_pct = 0.0060
    executor.trailing_tp_enabled = True
    executor.trailing_tp_giveback_pct = 0.35
    executor.trailing_tp_min_peak_pct = 0.0060
    executor.min_profit_after_fees = 0.0002
    return executor


def test_long_tp_scales_out_and_keeps_runner_open(tmp_path, monkeypatch):
    executor = _make_executor(tmp_path, monkeypatch)
    entry = 100.0
    pos = {
        "trade_id": 1,
        "side": "LONG",
        "entry": entry,
        "amount": 10.0,
        "entry_ts": time.time() - 60,
        "highest_price": entry,
        "lowest_price": entry,
        "highest_profit_pct": 0.0,
        "sl": 99.0,
        "tp_price": 100.6,
        "fee_rate": 0.0006,
        "min_profit_after_fees": 0.0002,
        "break_even_trigger_pct": 0.0060,
        "profit_trailing_enabled": True,
        "profit_trailing_activation_pct": 0.0060,
        "trailing_tp_enabled": True,
        "trailing_tp_giveback_pct": 0.35,
        "trailing_tp_min_peak_pct": 0.0060,
        "trail_tighten_1_pct": 0.0050,
        "trail_tighten_2_pct": 0.0100,
        "trail_t1_gap_pct": 0.0040,
        "trail_t2_gap_pct": 0.0030,
    }
    executor.active_positions = [pos]

    executor.process_orders_and_positions("BTC/USDT:USDT", 100.7)

    assert len(executor.active_positions) == 1
    assert executor.active_positions[0]["runner_scale_out_taken"] is True
    assert executor.active_positions[0]["profit_runner_armed"] is True
    assert executor.active_positions[0]["amount"] == pytest.approx(5.5)
    assert executor.active_positions[0]["sl"] >= entry * 1.0040
    assert executor.cash_usdt > 1000.0
    assert executor.stats_trades == 0


def test_long_runner_can_exit_after_partial_tp(tmp_path, monkeypatch):
    executor = _make_executor(tmp_path, monkeypatch)
    entry = 100.0
    pos = {
        "trade_id": 2,
        "side": "LONG",
        "entry": entry,
        "amount": 10.0,
        "entry_ts": time.time() - 60,
        "highest_price": entry,
        "lowest_price": entry,
        "highest_profit_pct": 0.0,
        "sl": 99.0,
        "tp_price": 100.6,
        "fee_rate": 0.0006,
        "min_profit_after_fees": 0.0002,
        "break_even_trigger_pct": 0.0060,
        "profit_trailing_enabled": True,
        "profit_trailing_activation_pct": 0.0060,
        "trailing_tp_enabled": True,
        "trailing_tp_giveback_pct": 0.35,
        "trailing_tp_min_peak_pct": 0.0060,
        "trail_tighten_1_pct": 0.0050,
        "trail_tighten_2_pct": 0.0100,
        "trail_t1_gap_pct": 0.0040,
        "trail_t2_gap_pct": 0.0030,
    }
    executor.active_positions = [pos]

    executor.process_orders_and_positions("BTC/USDT:USDT", 100.7)
    runner_sl = executor.active_positions[0]["sl"]

    executor.process_orders_and_positions("BTC/USDT:USDT", runner_sl - 0.01)

    assert executor.active_positions == []
    assert executor.trade_count == 1
