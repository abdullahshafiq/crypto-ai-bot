import ccxt
import os
import sys

def main():
    try:
        # Load from .env robustly
        env = {}
        if os.path.exists('.env'):
            with open('.env') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip()
        
        # Match the exact keys found in the previous run
        api_key = env.get('BINANCE_API_KEY')
        api_secret = env.get('BINANCE_SECRET') # Fixed from BINANCE_API_SECRET
        
        if not api_key or not api_secret:
            print("Error: Binance API keys not found in .env")
            print(f"Found keys: {list(env.keys())}")
            return

        exch = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'}
        })
        
        symbol = "SOL/USDC:USDC"
        positions = exch.fetch_positions([symbol])
        
        if not positions:
            print("No active positions found for SOL/USDC.")
            return

        pos = positions[0]
        entry = float(pos.get('entryPrice', 0.0))
        contracts = float(pos.get('contracts', 0.0))
        side = "LONG" if contracts > 0 else "SHORT"
        
        ticker = exch.fetch_ticker(symbol)
        current_price = ticker['last']
        
        pnl = (current_price - entry) * abs(contracts) if side == "LONG" else (entry - current_price) * abs(contracts)
        pnl_pct = (pnl / (entry * abs(contracts))) * 100 if entry != 0 else 0.0
        
        print(f"--- BINANCE LIVE AUDIT ---")
        print(f"Symbol:   {symbol}")
        print(f"Side:     {side}")
        print(f"Size:     {abs(contracts)} SOL")
        print(f"Entry:    ${entry:.4f}")
        print(f"Current:  ${current_price:.4f}")
        print(f"P&L:      ${pnl:+.4f} ({pnl_pct:+.2f}%)")
        print(f"--------------------------")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
