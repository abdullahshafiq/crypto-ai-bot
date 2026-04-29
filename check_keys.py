import ccxt
import os
from dotenv import load_dotenv

load_dotenv()


def check_keys():
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")

    print("Checking keys for Binance Futures Demo Trading...")
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "options": {"defaultType": "future"},
    })
    demo_base = "https://demo-fapi.binance.com"
    for k, url in list(exchange.urls.get("api", {}).items()):
        if not isinstance(url, str) or not k.startswith("fapi"):
            continue
        exchange.urls["api"][k] = url.replace("https://fapi.binance.com", demo_base)

    try:
        balance = exchange.fetch_balance()
        print("? SUCCESS: Keys are working for Binance Futures!")
        print(f"USDT Balance: {balance.get('USDT', {}).get('total', 0)}")
    except Exception as e:
        err = str(e)
        print(f"? ERROR: {err}")
        if "API-key format invalid" in err or "Invalid API-key" in err or "Invalid Api-Key ID" in err:
            print("\nNOTE: These keys are not valid for Binance Futures Demo Trading.")
            print("Use Demo Trading API keys for the demo-fapi environment (not Spot Testnet keys).")


if __name__ == "__main__":
    check_keys()
