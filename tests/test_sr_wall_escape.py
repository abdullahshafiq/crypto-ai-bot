import unittest

from safety import sr_wall_escape_ready


class SrWallEscapeTest(unittest.TestCase):
    def test_sell_breakdown_through_support_is_allowed_when_trend_is_already_bearish(self):
        self.assertTrue(
            sr_wall_escape_ready(
                action="SELL",
                sr_score=1.5,
                total_score=-0.45,
                mtf_fast_bias="SHORT_ONLY",
                macd_diff_val=-0.01,
                psar_bull=False,
                current_price=9.645,
                ema_9_val=9.688,
                breakout_gate=0.15,
                break_confirmed=True,
            )
        )

    def test_sell_near_support_is_still_blocked_without_breakdown_confirmation(self):
        self.assertFalse(
            sr_wall_escape_ready(
                action="SELL",
                sr_score=1.5,
                total_score=-0.10,
                mtf_fast_bias="NEUTRAL",
                macd_diff_val=0.0,
                psar_bull=True,
                current_price=9.645,
                ema_9_val=9.688,
                breakout_gate=0.15,
            )
        )

    def test_buy_into_resistance_blocked_without_break_confirmation(self):
        self.assertFalse(
            sr_wall_escape_ready(
                action="BUY",
                sr_score=-1.5,
                total_score=0.90,
                mtf_fast_bias="LONG_ONLY",
                macd_diff_val=0.01,
                psar_bull=True,
                current_price=10.10,
                ema_9_val=10.00,
                breakout_gate=0.20,
                break_confirmed=False,
            )
        )

    def test_buy_into_resistance_allowed_after_confirmed_breakout(self):
        self.assertTrue(
            sr_wall_escape_ready(
                action="BUY",
                sr_score=-1.5,
                total_score=0.90,
                mtf_fast_bias="LONG_ONLY",
                macd_diff_val=0.01,
                psar_bull=True,
                current_price=10.10,
                ema_9_val=10.00,
                breakout_gate=0.20,
                break_confirmed=True,
            )
        )

    def test_sell_into_support_blocked_without_break_confirmation(self):
        self.assertFalse(
            sr_wall_escape_ready(
                action="SELL",
                sr_score=1.5,
                total_score=-0.90,
                mtf_fast_bias="SHORT_ONLY",
                macd_diff_val=-0.01,
                psar_bull=False,
                current_price=9.90,
                ema_9_val=10.00,
                breakout_gate=0.20,
                break_confirmed=False,
            )
        )

    def test_sell_into_support_allowed_after_confirmed_breakdown(self):
        self.assertTrue(
            sr_wall_escape_ready(
                action="SELL",
                sr_score=1.5,
                total_score=-0.90,
                mtf_fast_bias="SHORT_ONLY",
                macd_diff_val=-0.01,
                psar_bull=False,
                current_price=9.90,
                ema_9_val=10.00,
                breakout_gate=0.20,
                break_confirmed=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
