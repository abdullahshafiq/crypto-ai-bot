"""Quick test: Can your API keys reach Binance LIVE Futures?"""
import os
from dotenv import load_dotenv
load_dotenv()

import ccxt

key = os.getenv("BINANCE_API_KEY", "")
secret = os.getenv("BINANCE_SECRET", "")

print(f"Key present: {bool(key)} (starts with: {key[:8]}...)")
print(f"Secret present: {bool(secret)} (starts with: {secret[:8]}...)")

# Test 1: LIVE Futures
print("\n--- Test 1: LIVE Futures (fapi.binance.com) ---")
try:
    ex = ccxt.binance({
        'apiKey': key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True},
    })
    balance = ex.fetch_balance()
    usdt = float(balance.get('USDT', {}).get('free', 0) or 0)
    usdc = float(balance.get('USDC', {}).get('free', 0) or 0)
    print(f"  SUCCESS! USDT: ${usdt:.2f}  USDC: ${usdc:.2f}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: LIVE Spot (to check if keys work at all)
print("\n--- Test 2: LIVE Spot (api.binance.com) ---")
try:
    ex2 = ccxt.binance({
        'apiKey': key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot', 'adjustForTimeDifference': True},
    })
    balance2 = ex2.fetch_balance()
    usdt2 = float(balance2.get('USDT', {}).get('free', 0) or 0)
    usdc2 = float(balance2.get('USDC', {}).get('free', 0) or 0)
    print(f"  SUCCESS! USDT: ${usdt2:.2f}  USDC: ${usdc2:.2f}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 3: Testnet Futures
print("\n--- Test 3: TESTNET Futures ---")
try:
    ex3 = ccxt.binance({
        'apiKey': key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True},
    })
    testnet_base = 'https://testnet.binancefuture.com'
    for k, url in list(ex3.urls.get('api', {}).items()):
        if isinstance(url, str) and k.startswith('fapi'):
            ex3.urls['api'][k] = url.replace('https://fapi.binance.com', testnet_base)
    balance3 = ex3.fetch_balance()
    usdt3 = float(balance3.get('USDT', {}).get('free', 0) or 0)
    print(f"  SUCCESS! USDT: ${usdt3:.2f}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDiagnosis: If Test 1 FAILED but Test 2 SUCCEEDED, enable 'Futures' on your Binance API key.")
print("If all tests FAILED, your API key or IP is blocked.")
