import sys
import time
import types
import unittest

if "ccxt" not in sys.modules:
    sys.modules["ccxt"] = types.ModuleType("ccxt")

from execution import PaperFuturesExecution


class ReversalFlipRegressionTest(unittest.TestCase):
    def _make_ttl_executor(self, fee_rate: float = 0.0004, ttl_seconds: int = 1):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=fee_rate,
            max_closed_trades=10,
        )
        executor.ttl_exit_seconds = ttl_seconds
        executor.min_profit_after_fees = 0.0
        executor._log_trade = lambda *args, **kwargs: None
        return executor

    def _make_psar_exit_executor(self, psar: float, psar_streak: int, side: str):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        executor.min_profit_after_fees = 0.0
        executor.psar_exit_min_streak = 2
        executor._current_psar = psar
        executor._current_psar_streak = psar_streak
        executor._log_trade = lambda *args, **kwargs: None
        executor.active_positions = [
            {
                "trade_id": 20 if side == "LONG" else 21,
                "side": side,
                "entry": 10.0,
                "amount": 1.0,
                "sl": 9.5 if side == "LONG" else 10.5,
                "entry_mode": "TREND",
                "entry_ts": time.time() - 61,
                "hold_until_ts": 0.0,
            }
        ]
        return executor

    def _make_defensive_cooldown_executor(self, cooldown_seconds: int = 180):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        executor.fixed_trade_usdt = 100.0
        executor.min_seconds_between_trades = 0
        executor.same_side_reentry_cooldown_seconds = cooldown_seconds
        executor.use_limit_orders = False
        executor.calculate_dynamic_leverage = lambda confidence, score, atr_pct=None: 1.0
        executor._log_trade = lambda *args, **kwargs: None
        executor._last_profitable_exit_side = ""
        executor._last_profitable_exit_ts = 0.0
        executor._last_defensive_exit_side = "SHORT"
        executor._last_defensive_exit_ts = time.time() - 60
        return executor

    def test_profitable_reversal_exits_and_reenters_opposite_side(self):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )

        executor.fixed_trade_usdt = 100.0
        executor.min_seconds_between_trades = 0
        executor.min_seconds_before_reversal = 0
        executor.reversal_min_confidence = 0.0
        executor.reversal_min_score = 0.0
        executor.reversal_min_net_edge_pct = 0.0
        executor.min_profit_after_fees = 0.0
        executor.same_side_reentry_cooldown_seconds = 0
        executor.same_side_reentry_strong_confidence = 0.0
        executor.use_limit_orders = False
        executor.calculate_dynamic_leverage = lambda confidence, score, atr_pct=None: 1.0
        executor._log_trade = lambda *args, **kwargs: None

        executor.active_positions = [
            {
                "trade_id": 1,
                "side": "SHORT",
                "entry": 10.0,
                "amount": 1.0,
                "entry_ts": 100.0,
                "hold_until_ts": 0.0,
            }
        ]

        executor.place_limit_order(
            {
                "action": "BUY",
                "confidence": 0.90,
                "score": 0.90,
                "tp": 9.8,
                "sl": 9.4,
                "reason": "reversal flip regression",
            },
            "AVAX/USDC:USDC",
            9.5,
        )

        self.assertEqual(len(executor.closed_trades), 1)
        self.assertEqual(executor.closed_trades[0]["type"], "REVERSAL_BANK")
        self.assertAlmostEqual(executor.closed_trades[0]["pnl"], 0.5, places=6)
        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "LONG")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY BUY"))

    def test_same_side_reentry_cooldown_blocks_short_but_not_opposite_long(self):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )

        executor.fixed_trade_usdt = 100.0
        executor.min_seconds_between_trades = 0
        executor.min_seconds_before_reversal = 0
        executor.reversal_min_confidence = 0.0
        executor.reversal_min_score = 0.0
        executor.reversal_min_net_edge_pct = 0.0
        executor.min_profit_after_fees = 0.0
        executor.same_side_reentry_cooldown_seconds = 300
        executor.same_side_reentry_strong_confidence = 0.0
        executor.use_limit_orders = False
        executor.calculate_dynamic_leverage = lambda confidence, score, atr_pct=None: 1.0
        executor._log_trade = lambda *args, **kwargs: None
        executor._last_profitable_exit_side = "SHORT"
        executor._last_profitable_exit_ts = __import__("time").time() - 120
        executor._opposite_reset_seen_after_profit = False

        executor.place_limit_order(
            {
                "action": "SELL",
                "confidence": 0.90,
                "score": 0.90,
                "tp": 9.8,
                "sl": 10.4,
                "reason": "same-side cooldown regression",
            },
            "AVAX/USDC:USDC",
            10.0,
        )

        self.assertEqual(len(executor.active_positions), 0)
        self.assertIn("same-side cooldown", executor.last_status)

        executor.place_limit_order(
            {
                "action": "BUY",
                "confidence": 0.90,
                "score": 0.90,
                "tp": 10.2,
                "sl": 9.6,
                "reason": "opposite-side regression",
            },
            "AVAX/USDC:USDC",
            10.0,
        )

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "LONG")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY BUY"))

    def test_long_flat_entry_within_grace_does_not_defensive_exit(self):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        executor._log_trade = lambda *args, **kwargs: None
        executor.active_positions = [
            {
                "trade_id": 2,
                "side": "LONG",
                "entry": 10.2,
                "amount": 1.0,
                "sl": 9.2,
                "entry_mode": "TREND",
                "entry_ts": time.time() - 119,
                "hold_until_ts": 0.0,
            }
        ]
        executor._current_signal_snapshot = {
            "mtf_fast_bias": "SHORT_ONLY",
            "structure_support": 9.9,
            "structure_resistance": 10.1,
        }

        executor.process_orders_and_positions("AVAX/USDC:USDC", 10.2)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(len(executor.closed_trades), 0)

    def test_short_flat_entry_within_grace_does_not_defensive_exit(self):
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        executor._log_trade = lambda *args, **kwargs: None
        executor.active_positions = [
            {
                "trade_id": 3,
                "side": "SHORT",
                "entry": 10.2,
                "amount": 1.0,
                "sl": 10.8,
                "entry_mode": "RANGE",
                "entry_ts": time.time() - 59,
                "hold_until_ts": 0.0,
            }
        ]
        executor._current_signal_snapshot = {
            "mtf_fast_bias": "LONG_ONLY",
            "structure_support": 9.9,
            "structure_resistance": 10.1,
        }

        executor.process_orders_and_positions("AVAX/USDC:USDC", 10.2)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(len(executor.closed_trades), 0)

    def test_short_adverse_below_40pct_sl_does_not_exit(self):
        short_executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        short_executor._log_trade = lambda *args, **kwargs: None
        short_executor.active_positions = [
            {
                "trade_id": 4,
                "side": "SHORT",
                "entry": 10.2,
                "amount": 1.0,
                "sl": 10.8,
                "entry_mode": "TREND",
                "entry_ts": time.time() - 121,
                "hold_until_ts": 0.0,
            }
        ]
        short_executor._current_signal_snapshot = {
            "mtf_fast_bias": "LONG_ONLY",
            "structure_resistance": 10.1,
            "structure_support": 9.9,
        }

        short_executor.process_orders_and_positions("AVAX/USDC:USDC", 10.4)

        self.assertEqual(len(short_executor.active_positions), 1)
        self.assertEqual(len(short_executor.closed_trades), 0)

    def test_long_adverse_above_40pct_sl_can_exit(self):
        long_executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=0.0,
            max_closed_trades=10,
        )
        long_executor._log_trade = lambda *args, **kwargs: None
        long_executor.active_positions = [
            {
                "trade_id": 5,
                "side": "LONG",
                "entry": 10.2,
                "amount": 1.0,
                "sl": 9.2,
                "entry_mode": "TREND",
                "entry_ts": time.time() - 121,
                "hold_until_ts": 0.0,
            }
        ]
        long_executor._current_signal_snapshot = {
            "mtf_fast_bias": "SHORT_ONLY",
            "structure_support": 10.1,
            "structure_resistance": 10.3,
        }

        long_executor.process_orders_and_positions("AVAX/USDC:USDC", 9.7)

        self.assertEqual(len(long_executor.active_positions), 0)
        self.assertEqual(long_executor.closed_trades[0]["type"], "DEFENSIVE_EXIT")
        self.assertTrue(long_executor.last_status.startswith("PAPER DEFENSIVE_EXIT"))

    def test_ttl_expired_profitable_trade_moves_sl_to_be_plus_fees_and_stays_open(self):
        executor = self._make_ttl_executor()
        executor.active_positions = [
            {
                "trade_id": 6,
                "side": "LONG",
                "entry": 10.0,
                "amount": 1.0,
                "sl": 9.5,
                "fee_rate": 0.0004,
                "min_profit_after_fees": 0.0,
                "entry_ts": time.time() - 3,
                "hold_until_ts": 0.0,
            }
        ]
        executor._current_signal_snapshot = {"mtf_fast_bias": "NEUTRAL"}

        executor.process_orders_and_positions("AVAX/USDC:USDC", 10.01)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(len(executor.closed_trades), 0)
        self.assertAlmostEqual(executor.active_positions[0]["sl"], 10.008, places=6)
        self.assertEqual(executor.stats_trades, 0)

    def test_ttl_expired_losing_trade_still_exits_and_updates_stats(self):
        executor = self._make_ttl_executor(fee_rate=0.0)
        executor.active_positions = [
            {
                "trade_id": 7,
                "side": "LONG",
                "entry": 10.0,
                "amount": 1.0,
                "sl": 9.5,
                "fee_rate": 0.0,
                "min_profit_after_fees": 0.0,
                "entry_ts": time.time() - 3,
                "hold_until_ts": 0.0,
            }
        ]
        executor._current_signal_snapshot = {"mtf_fast_bias": "NEUTRAL"}

        executor.process_orders_and_positions("AVAX/USDC:USDC", 9.95)

        self.assertEqual(len(executor.active_positions), 0)
        self.assertEqual(len(executor.closed_trades), 1)
        self.assertEqual(executor.closed_trades[0]["type"], "TTL_EXIT_LOSS")
        self.assertEqual(executor.stats_trades, 1)
        self.assertEqual(executor.stats_losses, 1)
        self.assertTrue(executor.last_status.startswith("PAPER TTL_EXIT_LOSS"))

    def test_one_candle_psar_flip_does_not_exit_long(self):
        executor = self._make_psar_exit_executor(psar=10.10, psar_streak=-1, side="LONG")

        executor.process_orders_and_positions("AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(len(executor.closed_trades), 0)

    def test_two_candle_bearish_psar_streak_exits_long(self):
        executor = self._make_psar_exit_executor(psar=10.10, psar_streak=-2, side="LONG")

        executor.process_orders_and_positions("AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 0)
        self.assertEqual(len(executor.closed_trades), 1)
        self.assertEqual(executor.closed_trades[0]["type"], "PSAR_EXIT")
        self.assertTrue(executor.last_status.startswith("PAPER PSAR_EXIT"))

    def test_one_candle_psar_flip_does_not_exit_short(self):
        executor = self._make_psar_exit_executor(psar=9.90, psar_streak=1, side="SHORT")

        executor.process_orders_and_positions("AVAX/USDC:USDC", 9.97)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(len(executor.closed_trades), 0)

    def test_two_candle_bullish_psar_streak_exits_short(self):
        executor = self._make_psar_exit_executor(psar=9.90, psar_streak=2, side="SHORT")

        executor.process_orders_and_positions("AVAX/USDC:USDC", 9.97)

        self.assertEqual(len(executor.active_positions), 0)
        self.assertEqual(len(executor.closed_trades), 1)
        self.assertEqual(executor.closed_trades[0]["type"], "PSAR_EXIT")
        self.assertTrue(executor.last_status.startswith("PAPER PSAR_EXIT"))

    def test_short_defensive_exit_blocks_new_short_during_cooldown(self):
        executor = self._make_defensive_cooldown_executor(cooldown_seconds=180)

        executor.place_limit_order(
            {
                "action": "SELL",
                "confidence": 0.90,
                "score": -0.90,
                "tp": 9.6,
                "sl": 10.4,
                "reason": "defensive cooldown regression",
            },
            "AVAX/USDC:USDC",
            10.0,
        )

        self.assertEqual(len(executor.active_positions), 0)
        self.assertIn("post-defensive same-side cooldown", executor.last_status)

    def test_short_defensive_exit_does_not_block_valid_long(self):
        executor = self._make_defensive_cooldown_executor(cooldown_seconds=180)

        executor.place_limit_order(
            {
                "action": "BUY",
                "confidence": 0.90,
                "score": 0.90,
                "tp": 10.4,
                "sl": 9.6,
                "reason": "defensive cooldown opposite-side regression",
            },
            "AVAX/USDC:USDC",
            10.0,
        )

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "LONG")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY BUY"))

    def test_defensive_cooldown_expiry_allows_same_side_again(self):
        executor = self._make_defensive_cooldown_executor(cooldown_seconds=180)
        executor._last_defensive_exit_ts = time.time() - 181

        executor.place_limit_order(
            {
                "action": "SELL",
                "confidence": 0.90,
                "score": -0.90,
                "tp": 9.6,
                "sl": 10.4,
                "reason": "defensive cooldown expiry regression",
            },
            "AVAX/USDC:USDC",
            10.0,
        )

        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "SHORT")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY SELL"))


class PaperScaleInRegressionTest(unittest.TestCase):
    def _make_executor(self, fee_rate: float = 0.0) -> PaperFuturesExecution:
        executor = PaperFuturesExecution(
            starting_balance_usdt=1000.0,
            leverage=1,
            fee_rate=fee_rate,
            max_closed_trades=10,
        )
        executor.fixed_trade_usdt = 100.0
        executor.min_seconds_between_trades = 0
        executor.same_side_reentry_cooldown_seconds = 0
        executor.scale_in_enabled = True
        executor.scale_in_max_steps = 2
        executor.scale_in_min_pnl_pct = 0.0020
        executor.scale_in_cooldown_seconds = 180
        executor.scale_in_position_pct = 0.5
        executor.scale_in_max_exposure_pct = 0.50
        executor.scale_in_wall_buffer_pct = 0.002
        executor.use_limit_orders = False
        executor.calculate_dynamic_leverage = lambda confidence, score, atr_pct=None: 1.0
        executor._log_trade = lambda *args, **kwargs: None
        executor._last_profitable_exit_side = ""
        executor._last_profitable_exit_ts = 0.0
        return executor

    def test_profitable_long_buy_adds_scale_in(self):
        executor = self._make_executor(fee_rate=0.001)
        executor.active_positions = [
            {
                "trade_id": 11,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 2)
        self.assertAlmostEqual(float(executor.active_positions[-1]["trade_usdt"]), 50.0, places=6)

    def test_scale_in_recomputes_combined_sl_within_cap(self):
        executor = self._make_executor(fee_rate=0.001)
        executor.max_structural_sl_pct = 0.01
        executor.active_positions = [
            {
                "trade_id": 111,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in sl recompute",
            "sl": 9.4,
            "structure_support": 9.7,
            "structure_resistance": 10.6,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 2)
        self.assertAlmostEqual(float(executor.active_positions[0]["sl"]), float(executor.active_positions[1]["sl"]), places=8)
        combined_amount = sum(float(p["amount"]) for p in executor.active_positions)
        combined_entry = sum(float(p["entry"]) * float(p["amount"]) for p in executor.active_positions) / combined_amount
        combined_sl = float(executor.active_positions[0]["sl"])
        combined_risk = abs(combined_entry - combined_sl) / combined_entry
        self.assertLessEqual(combined_risk, float(executor.max_structural_sl_pct) + 1e-12)

    def test_scale_in_blocked_if_combined_risk_too_high(self):
        executor = self._make_executor(fee_rate=0.001)
        executor.default_sl_pct = 0.0030
        executor.max_structural_sl_pct = 0.0010
        executor.active_positions = [
            {
                "trade_id": 112,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.97,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in risk cap veto",
            "structure_support": 9.7,
            "structure_resistance": 10.6,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertIn("combined risk", executor.last_status)

    def test_losing_long_does_not_scale_in(self):
        executor = self._make_executor()
        executor.active_positions = [
            {
                "trade_id": 12,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 9.97)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertIn("scale-in needs profit", executor.last_status)

    def test_long_near_resistance_does_not_scale_in(self):
        executor = self._make_executor()
        executor.active_positions = [
            {
                "trade_id": 13,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in regression",
            "structure_support": 9.7,
            "structure_resistance": 10.05,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.04)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertIn("near resistance", executor.last_status)

    def test_scale_in_cooldown_blocks_entry(self):
        executor = self._make_executor()
        executor.active_positions = [
            {
                "trade_id": 14,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 60,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            }
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.03)

        self.assertEqual(len(executor.active_positions), 1)
        self.assertIn("scale-in cooldown", executor.last_status)

    def test_scale_in_max_steps_blocks_third_entry(self):
        executor = self._make_executor()
        executor.active_positions = [
            {
                "trade_id": 15,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
            {
                "trade_id": 16,
                "side": "LONG",
                "entry": 10.02,
                "amount": 10.0,
                "sl": 9.82,
                "entry_ts": time.time() - 300,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "scale-in regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.05)

        self.assertEqual(len(executor.active_positions), 2)
        self.assertIn("scale-in max steps", executor.last_status)

    def test_reversal_closes_all_scale_in_legs(self):
        executor = self._make_executor(fee_rate=0.001)
        executor.active_positions = [
            {
                "trade_id": 21,
                "side": "LONG",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 9.8,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
            {
                "trade_id": 22,
                "side": "LONG",
                "entry": 10.02,
                "amount": 10.0,
                "sl": 9.82,
                "entry_ts": time.time() - 300,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
        ]
        signal = {
            "action": "SELL",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "reversal regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 10.50)

        self.assertEqual(len(executor.closed_trades), 2)
        self.assertTrue(all(t["type"] == "REVERSAL_BANK" for t in executor.closed_trades))
        self.assertAlmostEqual(sum(t["fees"] for t in executor.closed_trades), 0.4102, places=4)
        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "SHORT")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY SELL"))

    def test_reversal_closes_all_short_scale_in_legs(self):
        executor = self._make_executor(fee_rate=0.001)
        executor.active_positions = [
            {
                "trade_id": 23,
                "side": "SHORT",
                "entry": 10.0,
                "amount": 10.0,
                "sl": 10.2,
                "entry_ts": time.time() - 400,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
            {
                "trade_id": 24,
                "side": "SHORT",
                "entry": 10.02,
                "amount": 10.0,
                "sl": 10.22,
                "entry_ts": time.time() - 300,
                "hold_until_ts": 0.0,
                "trade_usdt": 100.0,
            },
        ]
        signal = {
            "action": "BUY",
            "confidence": 0.80,
            "score": 0.70,
            "reason": "reversal regression",
            "structure_support": 9.7,
            "structure_resistance": 10.5,
        }

        executor.place_limit_order(signal, "AVAX/USDC:USDC", 9.50)

        self.assertEqual(len(executor.closed_trades), 2)
        self.assertTrue(all(t["type"] == "REVERSAL_BANK" for t in executor.closed_trades))
        self.assertAlmostEqual(sum(t["fees"] for t in executor.closed_trades), 0.3902, places=4)
        self.assertEqual(len(executor.active_positions), 1)
        self.assertEqual(executor.active_positions[0]["side"], "LONG")
        self.assertTrue(executor.last_status.startswith("PAPER ENTRY BUY"))


if __name__ == "__main__":
    unittest.main()
