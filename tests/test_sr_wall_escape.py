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


if __name__ == "__main__":
    unittest.main()
