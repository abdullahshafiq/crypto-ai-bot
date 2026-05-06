
import os
import time
import ccxt
import yaml
from dotenv import load_dotenv
from execution import BinanceFuturesExecution

# Load environment variables
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET') # Fixed variable name

def test_limit():
    print("--- TESTING LIMIT ORDER ON BINANCE FUTURES ---")
    if not api_key or not api_secret:
        print("Missing BINANCE_API_KEY / BINANCE_SECRET. Aborting test.")
        return

    live_test = os.getenv("ALLOW_LIVE_TEST_ORDER") == "1"
    if live_test:
        print("LIVE mode enabled by ALLOW_LIVE_TEST_ORDER=1.")

    # Initialize executor
    executor = BinanceFuturesExecution(
        api_key=api_key,
        api_secret=api_secret,
        leverage=2,
        is_demo=not live_test
    )
    bal = executor.get_portfolio_value(0.10)
    print(f"Verified Account Balance: ${bal:,.2f}")
    # Keep this diagnostic order small; place_limit_order sizes from portfolio value.
    executor.get_portfolio_value = lambda _current_price: 10.0
    # Set required attributes that main.py usually sets
    executor.dynamic_leverage_enabled = False
    executor.fixed_trade_usdt = 7.0
    executor.leverage_min = 1.0
    executor.leverage_max = 2.0
    executor.leverage_confidence_levels = {0.05: 1.0, 0.5: 2.0}
    executor.leverage_use_score = False
    executor.atr_volatility_scaling = False
    executor.fee_rate = 0.0002 # 0.02% maker
    executor.fee_edge_multiplier = 2.0
    executor.tp_pct = 0.0030
    executor.fee_slippage_buffer_pct = 0.0001

    symbol = "DOGE/USDT:USDT"

    try:
        # 1. Fetch current price from order book
        order_book = executor.exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        print(f"Current Best Bid: {bid}")

        # 2. Set a price slightly BELOW the bid to ensure it stays as a Limit order
        test_price = bid * 0.99
        amount = 60 # display-only estimate; executor sizes from the patched test equity above

        print(f"Attempting to place LIMIT BUY order: {amount} DOGE @ {test_price:.5f}")

        # Mock a signal
        signal = {
            'action': 'BUY',
            'confidence': 0.5,
            'score': 0.5,
            'entry': test_price
        }

        # We call the method directly to see it work
        # Note: we use a tiny amount that clears the $5 minNotional
        executor.place_limit_order(signal, symbol, bid)

        print(f"Status Result: {executor.last_status}")

        if "Limit placed" in executor.last_status:
            print("\nSUCCESS! The limit order was accepted by Binance.")

            # 3. Cancel it immediately so we don't actually buy anything
            print("Cancelling test order...")
            executor.exchange.cancel_all_orders(symbol)
            print("Order cancelled. Test complete.")
        else:
            print(f"\nFAILED: {executor.last_status}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_limit()
