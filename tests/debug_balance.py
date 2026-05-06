import ccxt
import os
import json
from dotenv import load_dotenv

load_dotenv()

def debug_balance():
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")

    print("--- DEBUGGING BINANCE FUTURES BALANCE ---")
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'options': {'defaultType': 'future'}
    })
    # CCXT futures sandbox/testnet mode is deprecated; use Demo Trading endpoints.
    demo_base = 'https://demo-fapi.binance.com'
    for k, url in list(exchange.urls.get('api', {}).items()):
        if not isinstance(url, str) or not k.startswith('fapi'):
            continue
        exchange.urls['api'][k] = url.replace('https://fapi.binance.com', demo_base)

    try:
        balance = exchange.fetch_balance()
        print("RAW BALANCE keys found:", balance.keys())

        if 'USDT' in balance:
            print("USDT CCXT mapping:", balance['USDT'])

        if 'info' in balance:
            info = balance['info']
            print("\nRAW INFO snippet:")
            # Print only keys and a few values to avoid huge output
            for k, v in info.items():
                if 'balance' in k.lower() or 'equity' in k.lower() or 'asset' in k.lower():
                    print(f"  {k}: {v}")

            if 'assets' in info:
                for a in info['assets']:
                    if a['asset'] == 'USDT':
                        print(f"\nUSDT ASSET DATA: {a}")

    except Exception as e:
        print(f"ERROR DURING FETCH: {e}")

if __name__ == "__main__":
    debug_balance()
