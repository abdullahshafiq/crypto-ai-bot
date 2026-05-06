import sys
import types
import unittest

if "ccxt" not in sys.modules:
    sys.modules["ccxt"] = types.ModuleType("ccxt")

from execution import PaperFuturesExecution


class ReversalFlipRegressionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
