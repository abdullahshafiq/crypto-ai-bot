import os
import sys
import time
import yaml
import logging
import shutil
from collections import deque
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from market_data import MarketData
from indicators import calculate_base_indicators, generate_quant_signal, build_mtf_timeframe_context, compute_advanced_pivots
from news_data import NewsData
from agents import HybridAIOrchestrator
from execution import BinanceFuturesExecution, PaperFuturesExecution

_WIN_VT_ENABLED: bool | None = None
_ALT_SCREEN_ENTERED: bool = False
_CURSOR_HIDDEN: bool = False
_LAST_FRAME_LINES: int = 0

# --- Global Color Palette ---
use_ansi = True # Enabled for modern Windows/Linux
CYAN    = '\033[96m'
MAGENTA = '\033[95m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
GREEN   = '\033[92m'
RED     = '\033[91m'
BOLD    = '\033[1m'
RESET   = '\033[0m'
CL      = '\033[K' # Clear line

def _get_windows_console_handle():
    if os.name != "nt":
        return None
    try:
        import ctypes  # type: ignore
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return None
        return handle
    except Exception:
        return None


def _windows_cursor_home() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes  # type: ignore

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short),
                ("Top", ctypes.c_short),
                ("Right", ctypes.c_short),
                ("Bottom", ctypes.c_short),
            ]

        class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
            _fields_ = [
                ("dwSize", COORD),
                ("dwCursorPosition", COORD),
                ("wAttributes", ctypes.c_ushort),
                ("srWindow", SMALL_RECT),
                ("dwMaximumWindowSize", COORD),
            ]

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = _get_windows_console_handle()
        if handle is None:
            return False
        info = CONSOLE_SCREEN_BUFFER_INFO()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        home = COORD(0, info.srWindow.Top)
        return bool(kernel32.SetConsoleCursorPosition(handle, home))
    except Exception:
        return False


def _enable_windows_vt_mode() -> bool:
    """
    Best-effort enabling of ANSI escape sequences on Windows consoles.
    Returns True if VT processing was enabled, else False.
    """
    global _WIN_VT_ENABLED
    if _WIN_VT_ENABLED is not None:
        return _WIN_VT_ENABLED

    if os.name != "nt":
        _WIN_VT_ENABLED = True
        return True

    try:
        import ctypes  # type: ignore

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False

        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False

        _WIN_VT_ENABLED = True
        return True
    except Exception:
        _WIN_VT_ENABLED = False
        return False


def _enter_alt_screen_ansi():
    global _ALT_SCREEN_ENTERED, _CURSOR_HIDDEN
    if _ALT_SCREEN_ENTERED:
        return
    # Alternate screen buffer + hide cursor (prevents scrollback spam).
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()
    _ALT_SCREEN_ENTERED = True
    _CURSOR_HIDDEN = True


def _exit_alt_screen_ansi():
    global _ALT_SCREEN_ENTERED, _CURSOR_HIDDEN
    if not _ALT_SCREEN_ENTERED:
        return
    # Show cursor + leave alternate buffer.
    sys.stdout.write("\033[?25h\033[?1049l")
    sys.stdout.flush()
    _ALT_SCREEN_ENTERED = False
    _CURSOR_HIDDEN = False


def _hide_cursor_ansi():
    global _CURSOR_HIDDEN
    if _CURSOR_HIDDEN:
        return
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    _CURSOR_HIDDEN = True


def _show_cursor_ansi():
    global _CURSOR_HIDDEN
    if not _CURSOR_HIDDEN:
        return
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
    _CURSOR_HIDDEN = False

def setup_logging(logging_cfg: dict | None = None):
    # Only log to file so we don't mess up the clean terminal dashboard
    logging_cfg = logging_cfg or {}
    max_mb = float(logging_cfg.get("max_mb", 5))
    backups = int(logging_cfg.get("backups", 3))
    max_bytes = int(max_mb * 1024 * 1024)
    if max_bytes <= 0:
        max_bytes = 5 * 1024 * 1024
    if backups < 1:
        backups = 1

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler("bot.log", maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        ]
    )

def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _detect_ui_mode(cfg_mode: str) -> str:
    # 1. Non-interactive terminals (piping, IDE outputs) MUST fallback safely
    if not sys.stdout.isatty():
        return "cls" if os.name == "nt" else "plain"

    # 2. FORCE 'ansi' mode on all interactive terminals that support it.
    # This prevents the UI from breaking or scrolling even if the user
    # misconfigures 'ui.mode' in their config.yaml.
    if os.name == "nt":
        if _enable_windows_vt_mode():
            return "ansi"
        if _get_windows_console_handle() is not None:
            return "win"
        return "cls"

    return "ansi"


def print_dashboard(ticks, symbol, regime, state, signal, executor, session_start, mtf_context=None, mtf_cfg=None, status_lines=None, ui_cfg=None, ai_overlay=None, pivot_data=None, open_orders=None):
    global _LAST_FRAME_LINES
    ui_cfg = ui_cfg or {}
    ui_mode = _detect_ui_mode(str(ui_cfg.get("mode", "auto")))
    use_alt = bool(ui_cfg.get("alt_screen", False))

    if ui_mode == "ansi":
        # Force alternate screen buffer by default to prevent scrolling/repeated headers
        _enter_alt_screen_ansi()
        # Move cursor to top-left
        sys.stdout.write('\033[H')
    elif ui_mode == "win":
        _windows_cursor_home()
    elif ui_mode == "cls":
        os.system('cls' if os.name == 'nt' else 'clear')
    
    current_price = state['price']
    val = executor.get_portfolio_value(current_price)
    usdt = executor._fetch_free_usdt()
    btc = executor._fetch_free_btc()
    
    pnl = val - executor.initial_balance
    pnl_pct = (pnl / executor.initial_balance) * 100 if executor.initial_balance > 0 else 0
    uptime = time.time() - session_start
    
    m, s = divmod(int(uptime), 60)
    h, m = divmod(m, 60)
    
    ret_30s = state.get('ret_30s', 0)
    ret_30s_str = f"{ret_30s:+.2%}" if ret_30s is not None else "0.00%"

    # Enable ANSI colors — works on modern Windows PowerShell for all modes
    if os.name == "nt":
        _enable_windows_vt_mode()
        use_ansi = True
    else:
        use_ansi = (ui_mode in {"ansi", "plain", "cls"}) or (ui_mode == "win" and _enable_windows_vt_mode())
    
    term_width = shutil.get_terminal_size((100, 40)).columns
    frame_width = max(72, min(108, term_width - 2))
    compact = bool(ui_cfg.get("compact", True))

    # (Global Color Palette used)
    
    def colorize(val_num, text):
        if val_num > 0: return f"{GREEN}{text}{RESET}"
        elif val_num < 0: return f"{RED}{text}{RESET}"
        return text

    pnl_str = colorize(pnl, f"${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    
    # Action formatting
    action_color = GREEN if signal['action'] == "BUY" else (RED if signal['action'] == "SELL" else YELLOW)
    action_str = f"{action_color}{BOLD}{signal['action']}{RESET}"
    market_bias = str(signal.get("market_bias", "NEUTRAL") or "NEUTRAL").upper()
    bias_color = GREEN if market_bias == "LONG_ONLY" else (RED if market_bias == "SHORT_ONLY" else YELLOW)
    bias_str = f"{bias_color}{market_bias}{RESET}"
    overlay = ai_overlay if isinstance(ai_overlay, dict) else {}
    overlay_bias = str(overlay.get("bias", "NEUTRAL") or "NEUTRAL").upper()
    overlay_risk = str(overlay.get("risk_mode", "NORMAL") or "NORMAL").upper()
    overlay_style = str(overlay.get("entry_style", "MIXED") or "MIXED").upper()
    overlay_avoid = bool(overlay.get("avoid_new_entries", False))
    overlay_rationale = str(overlay.get("rationale", "") or "")
    overlay_bias_c = GREEN if overlay_bias == "LONG_ONLY" else (RED if overlay_bias == "SHORT_ONLY" else YELLOW)
    overlay_risk_c = RED if overlay_risk == "RISK_OFF" else (YELLOW if overlay_risk == "CAUTIOUS" else GREEN)
    
    # AI Regime formatting
    regime_color = GREEN if regime == "BULLISH" else (RED if regime == "BEARISH" else (MAGENTA if regime == "VOLATILE" else YELLOW))
    regime_str = f"{regime_color}{regime}{RESET}"

    # Helper to pad content to fixed width regardless of ANSI codes
    def ln(label, value, label2="", value2="", width=frame_width):
        # We manually build the line to avoid layout shifting
        left = f" {YELLOW}{label}:{RESET} {value}"
        right = ""
        if label2:
            right = f" {YELLOW}{label2}:{RESET} {value2}"
        
        # Estimate plain text length (labels are usually 5-10 chars, values vary)
        # To be safe, we just use the CL code to wipe the rest of the terminal line
        return f"{left}{'  ' + right if right else ''}{CL}"

    out = []
    def trim_text(text, max_len):
        text = str(text or "")
        return text if len(text) <= max_len else text[:max_len - 3] + "..."

    out.append(f"{CYAN}{BOLD}{'='*frame_width}{RESET}{CL}")
    exec_label = getattr(executor, "label", "EXEC")
    bal_str = f"{GREEN}${val:,.2f}{RESET}" if val >= executor.initial_balance else f"{RED}${val:,.2f}{RESET}"
    out.append(f" {BOLD}🚀 {exec_label} | {symbol} | {regime_str}{RESET} | Bal: {bal_str} | {datetime.now().strftime('%H:%M:%S')} | {pnl_str}{CL}")
    out.append(f"{CYAN}{BOLD}{'='*frame_width}{RESET}{CL}")
    
    # Combined Data & Technicals
    def nearest_levels(p, s, r):
        s = [l for l in s if isinstance(l,(int,float)) and l < p]
        r = [l for l in r if isinstance(l,(int,float)) and l > p]
        return max(s) if s else None, min(r) if r else None
    def fmt_level(pre, l):
        if l is None: return f"{pre}:-"
        d = (l - current_price)/current_price
        return f"{pre}:{d:+.1%}"

    out.append(f" {BLUE}{BOLD}[ MARKET & TECH ]{RESET}{CL}")
    out.append(ln("Price", f"${current_price:.5f}", "Spread", f"{state.get('spread_pct', 0):.4%}"))
    
    # Inline MTF if available
    if mtf_cfg.get("enabled", False) and isinstance(mtf_context, dict):
        for tf in mtf_cfg.get("timeframes", ["3m", "15m", "1h"]):
            entry = mtf_context.get(tf)
            if isinstance(entry, dict):
                trend = str(entry.get("trend", "NEUT")).upper()[:4]
                out.append(f"  {tf:<3}: {trend} | {fmt_level('S', nearest_levels(current_price, entry.get('support_levels',[]) or [], [])[0])} | {fmt_level('R', nearest_levels(current_price, [], entry.get('resistance_levels',[]) or [])[1])}{CL}")

    if pivot_data and pivot_data.get('classic'):
        pp = pivot_data['classic']['pp']
        pp_c = GREEN if current_price > pp else RED
        out.append(f"  {YELLOW}Pivot:{RESET} {pp_c}${pp:.5f}{RESET}  {RED}R1:{RESET}${pivot_data['classic']['r1']:.5f}  {GREEN}S1:{RESET}${pivot_data['classic']['s1']:.5f}{CL}")

    out.append(f"{'-'*frame_width}{CL}")
    out.append(f" {BLUE}{BOLD}[ STRATEGY & AI ]{RESET}{CL}")
    out.append(ln("Action", f"{action_str} ({signal.get('confidence',0):.1%})", "Bias", f"{bias_str} / {overlay_bias_c}{overlay_bias}{RESET}"))
    if overlay:
        out.append(f" {YELLOW}Thinking:{RESET} {trim_text(overlay_rationale, frame_width - 12)}{CL}")
        if overlay_avoid:
            out.append(f" {RED}{BOLD}!!! AI ADVISING NO NEW ENTRIES !!!{RESET}{CL}")

    reason = signal.get('reason', 'N/A')
    out.append(f" {YELLOW}Reason:{RESET} {trim_text(reason, frame_width - 10)}{CL}")
    if signal.get("action") == "HOLD" and str(signal.get("hold_reason", "")):
        out.append(f" {YELLOW}Hold:{RESET} {trim_text(signal.get('hold_reason',''), frame_width - 8)}{CL}")
        
    out.append(f"{'-'*frame_width}{CL}")
    # Open Orders
    if open_orders:
        out.append(f"{BLUE}{BOLD}[ OPEN ORDERS ]{RESET}{CL}")
        for o in open_orders[:3]: # Show top 3
            side_c = GREEN if o['side'] == 'BUY' else RED
            side_s = f"{side_c}{o['side']}{RESET}"
            out.append(f" {side_s} {o['amount']:.0f} @ ${o['price']:.5f} | {o['type']}{CL}")
        out.append(f"{CYAN}{'-'*frame_width}{RESET}{CL}")

    # Market Exposure
    out.append(f"{BLUE}{BOLD}[ MARKET EXPOSURE ]{RESET}{CL}")
    pending = getattr(executor, 'pending_orders', [])
    active = getattr(executor, 'active_positions', [])
    
    if not pending and not active:
        out.append(f"  No active exposure{CL}")
    else:
        for p_order in pending:
            p_side = f"{GREEN if p_order['side'] == 'BUY' else RED}{p_order['side']}{RESET}"
            out.append(f"  {YELLOW}PENDING:{RESET} {p_side} ${p_order['amount']*p_order['price']:.0f} @ ${p_order['price']:.5f}{CL}")
        for i, pos in enumerate(active, 1):
            side_c = GREEN if pos['side'] == 'LONG' else RED
            pnl_val = (current_price - pos['entry'] if pos['side']=='LONG' else pos['entry']-current_price)
            pnl_pct = pnl_val / pos['entry'] * 100 if pos['entry'] else 0
            val_pnl = pnl_val * pos['amount']
            pnl_c = GREEN if val_pnl >= 0 else RED
            out.append(f"  {side_c}{pos['side']}{RESET} ${pos['amount']*pos['entry']:.0f} @ ${pos['entry']:.5f} | {pnl_c}PnL:${val_pnl:+,.2f} ({pnl_pct:+.2f}%){RESET}{CL}")

    closed = getattr(executor, 'closed_trades', [])
    if closed:
        out.append(f"{'-'*frame_width}{CL}")
        out.append(f" {BLUE}{BOLD}[ PERFORMANCE ]{RESET}{CL}")
        wins = int(getattr(executor, "stats_wins", 0))
        total = int(getattr(executor, "stats_trades", 0))
        net_pnl = float(getattr(executor, "stats_gross", 0)) - float(getattr(executor, "stats_fees", 0))
        pnl_c = GREEN if net_pnl >= 0 else RED
        bal_c = GREEN if val >= executor.initial_balance else RED
        out.append(f"  {YELLOW}WR:{RESET} {wins}/{total} ({wins/total*100 if total else 0:.0f}%)  {YELLOW}Net:{RESET} {pnl_c}${net_pnl:+,.2f}{RESET}  {YELLOW}Bal:{RESET} {bal_c}${val:,.2f}{RESET}  {YELLOW}Fees:{RESET} ${float(getattr(executor, 'stats_fees', 0)):,.2f}{CL}")
        recent = list(closed)[-5:]
        recent.reverse()
        for t in recent:
            t_c = GREEN if t['pnl'] >= 0 else RED
            out.append(f"  {t_c}{t['type']:<14}{RESET} ${t['entry']:.5f}->${t['exit']:.5f}  {t_c}${t['pnl']:+,.2f}{RESET}{CL}")

    out.append(f"{CYAN}{BOLD}{'='*frame_width}{RESET}{CL}")
    out.append(f" {MAGENTA}Press Ctrl+C to safely exit.{RESET}{CL}")
    
    if status_lines:
        out.append(f"{'-'*frame_width}{CL}")
        out.append(f" {BLUE}{BOLD}[ STATUS ]{RESET}{CL}")
        max_lines = int(ui_cfg.get("status_lines", 3) or 3)
        if max_lines < 1:
            max_lines = 1
        lines = list(status_lines)[-max_lines:]
        for s in lines:  # already formatted strings
            out.append(f"  {trim_text(s, frame_width - 2)}{CL}")
        # Pad to fixed height so the box doesn't jitter when fewer lines exist
        for _ in range(max(0, max_lines - len(lines))):
            out.append(f"  {CL}")

    # Wipe entire frame and write
    frame_lines = len(out)
    if ui_mode == "win" and _LAST_FRAME_LINES > frame_lines:
        for _ in range(_LAST_FRAME_LINES - frame_lines):
            out.append(" " * frame_width)
        frame_lines = len(out)

    tail_clear = '\033[J' if use_ansi else ('\n' if ui_mode in {"cls", "plain"} else '')
    sys.stdout.write('\n'.join(out) + tail_clear)
    sys.stdout.flush()
    _LAST_FRAME_LINES = frame_lines

def run_hybrid_bot():
    _enable_windows_vt_mode()
    
    load_dotenv()

    cfg = load_config()
    setup_logging(cfg.get("logging", {}) or {})
    logger = logging.getLogger("main")
    symbol = cfg['symbol']
    mtf_cfg = cfg.get("mtf", {}) or {}
    ai_trade_cfg = cfg.get("ai_trade", {}) or {}
    ai_overlay_cfg = cfg.get("ai_overlay", {}) or {}
    ui_cfg = cfg.get("ui", {}) or {}
    mem_cfg = cfg.get("memory", {}) or {}
    auto_learning_cfg = cfg.get("auto_learning", {}) or {}

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    print("Booting up Hybrid Crypto AI Bot...")

    exec_cfg = cfg.get("execution", {}) or {}
    exec_mode = str(exec_cfg.get("mode", os.getenv("EXECUTION_MODE", "paper"))).strip().lower()
    leverage = int(exec_cfg.get("leverage", 5))
    paper_starting_usdt = float(exec_cfg.get("paper_starting_balance_usdt", 1000.0))
    fee_rate = float(exec_cfg.get("fee_rate", 0.0004))

    # Try to fetch real fee rate from Binance (works even in paper mode)
    if api_key and api_secret and "your_testnet" not in str(api_key):
        try:
            import ccxt
            _tmp_ex = ccxt.binanceusdm({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True}
            })
            fee_symbol = symbol
            if "/" in fee_symbol and ":" not in fee_symbol:
                base, quote = fee_symbol.split("/", 1)
                quote = quote.split(":", 1)[0]
                fee_symbol = f"{base}/{quote}:{quote}"
            _fee_info = _tmp_ex.fetch_trading_fee(fee_symbol)
            if _fee_info:
                real_maker = float(_fee_info.get('maker', 0) or 0)
                real_taker = float(_fee_info.get('taker', 0) or 0)
                if real_maker > 0:
                    fee_rate = real_maker  # Use maker fee (limit orders)
                    print(f"Fetched real Binance fees: maker={real_maker:.4%}, taker={real_taker:.4%} -> using {fee_rate:.4%}")
                elif real_taker > 0:
                    fee_rate = real_taker
                    print(f"Fetched real Binance fees: taker={real_taker:.4%} -> using {fee_rate:.4%}")
            del _tmp_ex
        except Exception as e:
            print(f"Could not fetch Binance fees ({e}), using config: {fee_rate:.4%}")
    fee_slippage_buffer_pct = float(exec_cfg.get("fee_slippage_buffer_pct", 0.0))
    fee_edge_multiplier = float(exec_cfg.get("fee_edge_multiplier", 1.0))
    min_seconds_between_trades = int(exec_cfg.get("min_seconds_between_trades", 0))
    min_seconds_before_reversal = int(exec_cfg.get("min_seconds_before_reversal", 0))
    reversal_min_confidence = float(exec_cfg.get("reversal_min_confidence", 0.0))
    reversal_min_score = float(exec_cfg.get("reversal_min_score", 0.0))
    reversal_min_net_edge_pct = float(exec_cfg.get("reversal_min_net_edge_pct", 0.0030))
    break_even_trigger_pct = float(exec_cfg.get("break_even_trigger_pct", 0.0015))
    break_even_buffer_pct = float(exec_cfg.get("break_even_buffer_pct", 0.0002))
    trail_tighten_1_pct = float(exec_cfg.get("trail_tighten_1_pct", 0.0030))
    trail_tighten_2_pct = float(exec_cfg.get("trail_tighten_2_pct", 0.0060))
    min_profit_after_fees = float(exec_cfg.get("min_profit_after_fees", 0.0010))
    exit_on_reversal_only_in_profit = bool(exec_cfg.get("exit_on_reversal_only_in_profit", True))
    use_limit_orders = bool(exec_cfg.get("use_limit_orders", False))
    trailing_callback_pct = float(exec_cfg.get("trailing_callback_pct", 0.5))
    trailing_stop_callback = trailing_callback_pct / 100.0

    strategy_config = {
        'max_spread': cfg['strategy']['max_spread'],
        'min_conf': cfg['strategy']['min_conf'],
        'fixed_trade_usdt': cfg['strategy'].get('fixed_trade_usdt', 100),
        'tp_pct': cfg['strategy']['tp_pct'],
        'sl_pct': cfg['strategy']['sl_pct'],
    }
    fixed_trade_usdt = float(strategy_config.get('fixed_trade_usdt', 0.0) or 0.0)

    indicator_refresh_interval = cfg['intervals']['indicator_refresh']
    regime_refresh_interval = cfg['intervals']['regime_refresh']
    tick_delay = cfg['intervals']['tick_delay_seconds']
    ai_enabled = cfg['ai']['enabled']
    ai_model = cfg['ai']['model']
    
    data_cfg = cfg.get("data", {}) or {}
    market = MarketData(market=str(data_cfg.get("market", "usdm")))
    news = NewsData()
    ai_orch = HybridAIOrchestrator(model=ai_model)
    
    print("Fetching historical data to warm up indicators (EMAs/RSI/VWAP)...")
    hist_df = market.fetch_ohlcv(symbol, timeframe=cfg['timeframe'], limit=100)
    if hist_df.empty:
        # Fallback: try the other market type once (spot <-> usdm) for convenience.
        alt_market = "spot" if getattr(market, "market", "usdm") == "usdm" else "usdm"
        alt = MarketData(market=alt_market)
        alt_df = alt.fetch_ohlcv(symbol, timeframe=cfg['timeframe'], limit=100)
        if not alt_df.empty:
            market = alt
            hist_df = alt_df
            print(f"Note: switched market data source to `{alt_market}` for symbol {symbol}.")
        else:
            print(f"CRITICAL ERROR: Failed to fetch initial OHLCV data for {symbol} ({cfg['timeframe']}).")
            print("Tip: check symbol spelling and whether it exists on Futures (usdm) vs Spot.")
            print("Tip: try `symbol: \"BTC/USDT\"` to verify connectivity.")
            logger.critical("Failed to fetch initial OHLCV data. Exiting.")
            return

    df_indicators = calculate_base_indicators(hist_df)
    latest_indicators = df_indicators.iloc[-1].to_dict()
    bootstrap_price = float(hist_df.iloc[-1]['close'])

    executor = None
    if exec_mode in {"demo", "binance", "live"}:
        if not api_key or not api_secret or "your_testnet" in api_key:
            logger.warning("Binance keys missing/placeholder; falling back to PAPER execution.")
            print(f"\n{RED}{BOLD}[!] WARNING: Missing Binance API keys. Falling back to PAPER mode.{RESET}\n")
            exec_mode = "paper"
        else:
            is_demo_mode = (exec_mode != "live")
            
            mode_color = RED if not is_demo_mode else YELLOW
            mode_text = "LIVE (REAL MONEY)" if not is_demo_mode else "DEMO (TESTNET)"
            print(f"\n{mode_color}{BOLD}=== ENVIRONMENT CHECK ==={RESET}")
            print(f"Mode: {mode_color}{BOLD}{mode_text}{RESET}")
            print(f"Exchange: BINANCE FUTURES")
            print(f"Connecting to API to verify keys...{RESET}")
            executor = BinanceFuturesExecution(
                api_key,
                api_secret,
                leverage=leverage,
                max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
                is_demo=is_demo_mode
            )
            executor.symbol = symbol # Fix: Ensure symbol is set BEFORE balance check
            executor.fee_rate = fee_rate
            executor.fee_slippage_buffer_pct = fee_slippage_buffer_pct
            executor.fee_edge_multiplier = fee_edge_multiplier
            executor.fixed_trade_usdt = fixed_trade_usdt
            executor.min_seconds_between_trades = min_seconds_between_trades
            executor.min_seconds_before_reversal = min_seconds_before_reversal
            executor.use_limit_orders = use_limit_orders
            executor.reversal_min_confidence = reversal_min_confidence
            executor.reversal_min_score = reversal_min_score
            executor.reversal_min_net_edge_pct = reversal_min_net_edge_pct
            executor.break_even_trigger_pct = break_even_trigger_pct
            executor.break_even_buffer_pct = break_even_buffer_pct
            executor.trail_tighten_1_pct = trail_tighten_1_pct
            executor.trail_tighten_2_pct = trail_tighten_2_pct
            executor.min_profit_after_fees = min_profit_after_fees
            executor.exit_on_reversal_only_in_profit = exit_on_reversal_only_in_profit
            executor.use_native_trailing_stop = bool(cfg.get('execution', {}).get('use_native_trailing_stop', False))
            executor.trailing_callback_pct = trailing_callback_pct
            executor.trailing_stop_callback = trailing_stop_callback

            _ = executor.get_portfolio_value(bootstrap_price)
            if executor.initial_balance <= 0:
                logger.warning(f"Demo execution unavailable ({getattr(executor, 'last_status', '')}); falling back to PAPER.")
                exec_mode = "paper"
                executor = None

    if exec_mode == "paper" or executor is None:
        print(f"Initializing PAPER execution layer (starting ${paper_starting_usdt:,.2f})...")
        executor = PaperFuturesExecution(
            starting_balance_usdt=paper_starting_usdt,
            leverage=leverage,
            fee_rate=fee_rate,
            max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
        )
        executor.fee_slippage_buffer_pct = fee_slippage_buffer_pct
        executor.fee_edge_multiplier = fee_edge_multiplier
        executor.fixed_trade_usdt = fixed_trade_usdt
        executor.min_seconds_between_trades = min_seconds_between_trades
        executor.min_seconds_before_reversal = min_seconds_before_reversal
        executor.reversal_min_confidence = reversal_min_confidence
        executor.reversal_min_score = reversal_min_score
        executor.reversal_min_net_edge_pct = reversal_min_net_edge_pct
        executor.break_even_trigger_pct = break_even_trigger_pct
        executor.break_even_buffer_pct = break_even_buffer_pct
        executor.trail_tighten_1_pct = trail_tighten_1_pct
        executor.trail_tighten_2_pct = trail_tighten_2_pct
        executor.min_profit_after_fees = min_profit_after_fees
        executor.exit_on_reversal_only_in_profit = exit_on_reversal_only_in_profit
        executor.use_limit_orders = use_limit_orders

    executor.max_open_positions = cfg['risk'].get('max_open_positions', 1)
    executor.daily_loss_cap_pct = cfg['risk'].get('daily_loss_cap')
    executor.min_balance_floor = float(cfg['risk'].get('min_balance_floor', 90.0))

    auto_learning_enabled = bool(auto_learning_cfg.get("enabled", False))
    auto_learning_min_trades = int(auto_learning_cfg.get("min_completed_trades", 30))
    auto_learning_refresh_trades = max(1, int(auto_learning_cfg.get("refresh_closed_trades", 10)))
    auto_learning_max_recent = int(auto_learning_cfg.get("max_recent_trades", 300))
    auto_learning_shrinkage = float(auto_learning_cfg.get("shrinkage", 0.35))
    auto_learning_ai_cfg = auto_learning_cfg.get("ai_advisor", {}) or {}
    auto_learning_ai_enabled = bool(auto_learning_ai_cfg.get("enabled", False))
    auto_learning_ai_model = str(auto_learning_ai_cfg.get("model", ai_model))
    auto_learning_ai_max_shift = float(auto_learning_ai_cfg.get("max_weight_shift", 0.12))

    # Configure dynamic leverage if enabled
    leverage_cfg = cfg.get('leverage', {})
    executor.dynamic_leverage_enabled = bool(leverage_cfg.get('enabled', False))
    executor.leverage_min = float(leverage_cfg.get('min_leverage', 1.0))
    executor.leverage_max = float(leverage_cfg.get('max_leverage', 4.0))
    executor.leverage_use_score = bool(leverage_cfg.get('use_score_multiplier', False))
    executor.leverage_score_weight = float(leverage_cfg.get('score_weight', 0.3))
    executor.atr_volatility_scaling = bool(leverage_cfg.get('atr_volatility_scaling', False))
    executor.atr_reference_pct = float(leverage_cfg.get('atr_reference_pct', 0.02))
    executor.atr_min_multiplier = float(leverage_cfg.get('atr_min_multiplier', 0.3))
    
    # Load confidence level mapping
    conf_levels = leverage_cfg.get('confidence_levels', {})
    executor.leverage_confidence_levels = {float(k): float(v) for k, v in conf_levels.items()}
    
    # Configure DCA if enabled
    exec_cfg = cfg.get('execution', {})
    executor.dca_enabled = bool(exec_cfg.get('dca_enabled', False))
    executor.dca_max_steps = int(exec_cfg.get('dca_max_steps', 0))
    executor.dca_distance_pct = float(exec_cfg.get('dca_distance_pct', 0.01))
    
    if executor.dynamic_leverage_enabled:
        logger.info(f"Dynamic Leverage ENABLED: {executor.leverage_min:.1f}x-{executor.leverage_max:.1f}x (confidence-based)")

    if auto_learning_enabled:
        try:
            from ml_optimizer import optimize_weights

            learning_state = optimize_weights(
                min_trades=auto_learning_min_trades,
                max_recent_trades=auto_learning_max_recent,
                shrinkage=auto_learning_shrinkage,
                ai_enabled=auto_learning_ai_enabled,
                ai_model=auto_learning_ai_model,
                ai_max_weight_shift=auto_learning_ai_max_shift,
                quiet=True,
            )
            if learning_state:
                executor.learning_risk_multiplier = float(learning_state.get("risk_multiplier", 1.0) or 1.0)
                logger.info(
                    "Auto-learning initialized: %s trades, win_rate=%.1f%%, risk_multiplier=%.2f",
                    learning_state.get("completed_trades", 0),
                    float(learning_state.get("win_rate", 0.0)) * 100.0,
                    float(getattr(executor, "learning_risk_multiplier", 1.0)),
                )
        except Exception as e:
            logger.warning(f"Auto-learning startup skipped: {e}")

    regime = "NEUTRAL"
    if ai_enabled:
        print("AI Agents are reading the news to determine macro regime...")
        headlines = news.fetch_latest_news(symbol)
        regime = ai_orch.determine_macro_regime(
            headlines,
            f"Price: {hist_df.iloc[-1]['close']:.2f}, RSI: {latest_indicators.get('rsi_14', 'N/A')}"
        )

    print("Startup sequence complete. Entering high-frequency loop...")
    time.sleep(1) # Give user a second to read startup logs
    
    # Clear screen once before loop starts
    os.system('cls' if os.name == 'nt' else 'clear')

    ticks = 0
    state = None
    session_start = time.time()
    ticks = 0
    status_buf = deque(maxlen=20)
    last_reported_status = ""
    open_orders_cache = []
    mtf_context = {}
    latest_macro = None
    pivot_data = {}
    last_pivot_refresh_ts = 0.0
    df_indicators = None
    latest_indicators = {}
    last_ai_trade_ts = 0.0
    last_ai_trade_key = None
    last_ai_trade_resp = None
    last_reported_status = ""
    last_reported_signal = ""
    last_learning_closed_trades = int(getattr(executor, "stats_trades", 0) or 0)
    ai_overlay_state = {
        "bias": "NEUTRAL",
        "risk_mode": "NORMAL",
        "entry_style": "MIXED",
        "avoid_new_entries": False,
        "max_hold_minutes": 0,
        "confidence": 0.0,
        "rationale": "Overlay disabled",
        "computed_at": 0.0,
    }

    # Keep this bounded so long runs don't grow memory (and we only show the last few anyway).
    status_buf = deque(maxlen=80)
    signal_history = deque(maxlen=5) # 5-second smoothing buffer

    def status(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        status_buf.append(f"{ts} {msg}")

    try:
        while True:
            ticks += 1

            if ticks == 1 or ticks % indicator_refresh_interval == 0:
                logger.info("Refreshing macro indicators...")
                status("Refreshing indicators/MTF")
                new_df = market.fetch_ohlcv(symbol, timeframe=cfg['timeframe'], limit=100)
                macro_df = market.fetch_ohlcv(symbol, timeframe=cfg.get('macro_timeframe', '1h'), limit=100)

                if mtf_cfg.get("enabled", False):
                    for tf in mtf_cfg.get("timeframes", ["15m", "3h", "4h"]):
                        htf_df = market.fetch_ohlcv(symbol, timeframe=tf, limit=200)
                        if htf_df.empty:
                            status(f"MTF {tf}: fetch failed")
                            continue
                        htf_ind = calculate_base_indicators(htf_df)
                        ctx = build_mtf_timeframe_context(htf_ind)
                        ctx["computed_at"] = time.time()
                        mtf_context[str(tf)] = ctx
                    status("MTF context updated")
                
                if not new_df.empty:
                    df_indicators = calculate_base_indicators(new_df)
                    latest_indicators = df_indicators.iloc[-1].to_dict()
                    
                if not macro_df.empty:
                    macro_indicators_df = calculate_base_indicators(macro_df)
                    latest_macro = macro_indicators_df.iloc[-1].to_dict()

                # Refresh Advanced Pivot Points from daily OHLCV (every 15 min)
                if time.time() - last_pivot_refresh_ts >= 900 or not pivot_data:
                    try:
                        daily_df = market.fetch_ohlcv(symbol, timeframe='1d', limit=5)
                        if not daily_df.empty and len(daily_df) >= 2:
                            pivot_data = compute_advanced_pivots(daily_df)
                            last_pivot_refresh_ts = time.time()
                            classic_pp = pivot_data.get('classic', {})
                            if classic_pp:
                                status(f"Pivots: PP={classic_pp['pp']:.5f} S1={classic_pp['s1']:.5f} R1={classic_pp['r1']:.5f}")
                    except Exception as e:
                        logger.warning(f"Failed to compute pivots: {e}")

            if ai_enabled and (ticks == 1 or ticks % regime_refresh_interval == 0):
                logger.info("Refreshing AI macro regime...")
                headlines = news.fetch_latest_news(symbol)
                funding_rate = market.fetch_funding_rate(symbol)
                
                quant_context = (
                    f"Price: {latest_indicators.get('close', 'N/A')}\n"
                    f"RSI: {latest_indicators.get('rsi_14', 'N/A')}\n"
                    f"ADX (Trend Strength): {latest_indicators.get('adx', 'N/A')}\n"
                    f"Futures Funding Rate: {funding_rate:.4%}"
                )
                
                regime = ai_orch.determine_macro_regime(headlines, quant_context)

            overlay_interval = int(ai_overlay_cfg.get("refresh_seconds", 1800) or 1800)
            overlay_enabled = bool(ai_overlay_cfg.get("enabled", False))
            if overlay_enabled and (time.time() - float(ai_overlay_state.get("computed_at", 0.0) or 0.0) >= max(60, overlay_interval)):
                status("Refreshing AI overlay")
                headlines = news.fetch_latest_news(symbol)
                overlay_context = {
                    "symbol": symbol,
                    "regime": regime,
                    "timeframe": cfg.get("timeframe"),
                    "macro_timeframe": cfg.get("macro_timeframe"),
                    "price": state.get("price") if isinstance(state, dict) else latest_indicators.get("close"),
                    "ret_30s": state.get("ret_30s") if isinstance(state, dict) else None,
                    "volume_state": state.get("volume_state") if isinstance(state, dict) else None,
                    "latest_indicators": {
                        "rsi_14": latest_indicators.get("rsi_14"),
                        "adx": latest_indicators.get("adx"),
                        "ema_9": latest_indicators.get("ema_9"),
                        "ema_21": latest_indicators.get("ema_21"),
                    },
                    "mtf_context": mtf_context,
                    "mtf_summary": {tf: (mtf_context.get(tf, {}) or {}).get("trend", "N/A") for tf in ["3m", "5m", "15m"]},
                    "headlines": headlines[:8],
                    "symbol_info": symbol
                }
                overlay_model = str(ai_overlay_cfg.get("model", ai_model))
                ai_overlay_state = ai_orch.evaluate_overlay(overlay_context, model=overlay_model)
                ai_overlay_state["computed_at"] = time.time()
                status(f"AI overlay: {ai_overlay_state.get('bias','NEUTRAL')} / {ai_overlay_state.get('risk_mode','NORMAL')}")

            state = market.fetch_order_book_and_ticks(symbol)
            if state is None:
                status("Market data: retrying (API/ratelimit)")
                time.sleep(tick_delay)
                continue

            # Update executor with current ATR for volatility-based leverage
            atr_pct = latest_indicators.get('atr_pct')
            if atr_pct is not None and not (isinstance(atr_pct, float) and atr_pct != atr_pct):  # Check not NaN
                executor._current_atr_pct = float(atr_pct)

            executor.process_orders_and_positions(symbol, state['price'])

            if auto_learning_enabled:
                closed_trades = int(getattr(executor, "stats_trades", 0) or 0)
                if closed_trades >= last_learning_closed_trades + auto_learning_refresh_trades:
                    try:
                        from ml_optimizer import optimize_weights

                        learning_state = optimize_weights(
                            min_trades=auto_learning_min_trades,
                            max_recent_trades=auto_learning_max_recent,
                            shrinkage=auto_learning_shrinkage,
                            ai_enabled=auto_learning_ai_enabled,
                            ai_model=auto_learning_ai_model,
                            ai_max_weight_shift=auto_learning_ai_max_shift,
                            quiet=True,
                        )
                        last_learning_closed_trades = closed_trades
                        if learning_state:
                            executor.learning_risk_multiplier = float(learning_state.get("risk_multiplier", 1.0) or 1.0)
                            wr = float(learning_state.get("win_rate", 0.0)) * 100.0
                            risk_mult = float(getattr(executor, "learning_risk_multiplier", 1.0))
                            status(f"Auto-learning updated weights (WR {wr:.1f}%, risk {risk_mult:.2f}x)")
                            logger.info(f"Auto-learning updated weights: {learning_state}")
                    except Exception as e:
                        last_learning_closed_trades = closed_trades
                        logger.warning(f"Auto-learning update skipped: {e}")

            if not executor.check_risk_limits(state['price']):
                logger.critical("RISK LIMIT HIT. Halting all trading.")
                print(f"\n⛔ BOT HALTED: Balance dropped to ${executor.min_balance_floor:,.2f} floor OR daily loss cap exceeded.")
                print("   All positions will be liquidated for safety.")
                executor.close_all_positions(symbol)
                break

            signal = generate_quant_signal(
                state,
                latest_indicators,
                strategy_config,
                df_indicators,
                latest_macro,
                mtf_context=mtf_context,
                mtf_config=mtf_cfg,
                pivot_data=pivot_data,
            )
            
            # SIGNAL SMOOTHING: Avoid reacting to 1s noise.
            # We average the last 5 seconds of scores.
            raw_score = float(signal.get('score', 0.0) or 0.0)
            signal_history.append(raw_score)
            avg_score = sum(signal_history) / len(signal_history)
            
            # If the smoothed score flips sign or is significantly weaker than the raw tick, we hold.
            if len(signal_history) >= 3:
                # If raw score and avg score have different signs, the move is jittery.
                if (raw_score > 0 and avg_score < 0) or (raw_score < 0 and avg_score > 0):
                    signal['action'] = "HOLD"
                    signal['hold_reason'] = "Signal smoothing: jitter detected"
                
                # Apply the smoothed score back to the signal
                signal['score'] = avg_score
                # Keep confidence as a 0.0 - 1.0 decimal for consistent UI formatting
                capped_score = max(-1.0, min(1.0, avg_score))
                signal['confidence'] = abs(capped_score)
                
                # If confidence is too low after smoothing, hold.
                if signal['confidence'] < float(strategy_config.get('min_conf', 0.05)):
                    signal['action'] = "HOLD"
                    signal['hold_reason'] = "Signal smoothing: weak confidence"

            sig_str = f"Signal: {signal.get('action','?')} conf={float(signal.get('confidence',0.0) or 0.0):.1%} Reason: {signal.get('reason','N/A')}"
            if sig_str != last_reported_signal:
                status(sig_str)
                logger.info(f"ANALYSIS: {sig_str}")
                last_reported_signal = sig_str

            if bool(ai_overlay_cfg.get("enabled", False)):
                overlay_bias = str(ai_overlay_state.get("bias", "NEUTRAL") or "NEUTRAL").upper()
                overlay_risk = str(ai_overlay_state.get("risk_mode", "NORMAL") or "NORMAL").upper()
                overlay_avoid = bool(ai_overlay_state.get("avoid_new_entries", False))
                overlay_hold_minutes = int(ai_overlay_state.get("max_hold_minutes", 0) or 0)
                overlay_note = str(ai_overlay_state.get("rationale", "") or "")[:120]

                # AI EMERGENCY EXIT: If we are counter-trend to AI Bias, liquidation is mandatory
                active_positions = getattr(executor, 'active_positions', [])
                if active_positions:
                    current_pos = active_positions[0]
                    if overlay_bias == "SHORT_ONLY" and current_pos['side'] == "LONG":
                        signal["action"] = "HOLD"
                        status("AI EMERGENCY: Liquidating LONG (Bias: SHORT_ONLY)")
                        executor.close_all_positions(symbol)
                    elif overlay_bias == "LONG_ONLY" and current_pos['side'] == "SHORT":
                        signal["action"] = "HOLD"
                        status("AI EMERGENCY: Liquidating SHORT (Bias: LONG_ONLY)")
                        executor.close_all_positions(symbol)

                if overlay_avoid and signal.get("action") in {"BUY", "SELL"}:
                    signal["action"] = "HOLD"
                    signal["hold_reason"] = "AI overlay: no new entries"
                    signal["reason"] = f"{signal.get('reason','')} [AI Overlay] {overlay_note}"
                elif overlay_bias == "LONG_ONLY" and signal.get("action") == "SELL":
                    signal["action"] = "HOLD"
                    signal["hold_reason"] = "AI overlay long-only: sell disabled"
                    signal["reason"] = f"{signal.get('reason','')} [AI Overlay] {overlay_note}"
                elif overlay_bias == "SHORT_ONLY" and signal.get("action") == "BUY":
                    signal["action"] = "HOLD"
                    signal["hold_reason"] = "AI overlay short-only: buy disabled"
                    signal["reason"] = f"{signal.get('reason','')} [AI Overlay] {overlay_note}"

                if signal.get("action") in {"BUY", "SELL"} and overlay_hold_minutes > 0:
                    signal["hold_until_ts"] = time.time() + (overlay_hold_minutes * 60)

            # Legacy trade-gating AI (evaluates each BUY/SELL before execution)
            if ai_trade_cfg.get("enabled", False) and signal.get("action") in {"BUY", "SELL"}:
                now = time.time()
                ai_resp = None
                use_cached = False
                max_hold_minutes = int(ai_trade_cfg.get("max_hold_minutes", 60))
                on_error = str(ai_trade_cfg.get("on_error", "allow")).strip().lower()
                current_pos = executor.active_positions[0] if getattr(executor, "active_positions", []) else None
                is_reversal = False
                if isinstance(current_pos, dict):
                    if (signal["action"] == "BUY" and current_pos.get("side") == "SHORT") or (signal["action"] == "SELL" and current_pos.get("side") == "LONG"):
                        is_reversal = True
                    else:
                        # We already have a position in this direction, don't ask AI again
                        pass

                # SKIP AI evaluation if we are already in the right direction
                should_skip_ai = False
                if current_pos and not is_reversal:
                    should_skip_ai = True
                
                if not should_skip_ai:
                    min_ivl = int(ai_trade_cfg.get("min_interval_seconds", 30))
                    # ULTRA-SIMPLE KEY: Only the action matters. Don't re-ask if signal hasn't flipped.
                    key = str(signal.get("action"))
                    use_cached = (last_ai_trade_key == key) and (last_ai_trade_resp is not None) and ((now - last_ai_trade_ts) < float(min_ivl))
                    
                    if use_cached:
                        ai_resp = last_ai_trade_resp
                    else:
                        status("Asking AI to evaluate trade...")
                        ai_model_trade = str(ai_trade_cfg.get("model", ai_model))

                        ctx = {
                            "symbol": symbol,
                            "mode": getattr(executor, "label", ""),
                            "proposed_action": signal.get("action"),
                            "is_reversal": is_reversal,
                            "price": state.get("price"),
                            "spread_pct": state.get("spread_pct"),
                            "ret_30s": state.get("ret_30s"),
                            "signal": {
                                "score": signal.get("score"),
                                "confidence": signal.get("confidence"),
                                "tp": signal.get("tp"),
                                "sl": signal.get("sl"),
                                "reason": str(signal.get("reason", ""))[:200],
                            },
                            "fees": {
                                "fee_rate_per_side": getattr(executor, "fee_rate", None),
                                "fee_slippage_buffer_pct": getattr(executor, "fee_slippage_buffer_pct", None),
                                "fee_edge_multiplier": getattr(executor, "fee_edge_multiplier", None),
                            },
                            "mtf": mtf_context,
                            "position": current_pos or None,
                        }
                        ai_resp = ai_orch.evaluate_trade(ctx, model=ai_model_trade)
                        last_ai_trade_ts = now
                        last_ai_trade_key = key
                        last_ai_trade_resp = ai_resp

                decision = str((ai_resp or {}).get("decision", "ALLOW")).upper()
                hold_minutes = int((ai_resp or {}).get("hold_minutes", 0) or 0)
                if hold_minutes < 0:
                    hold_minutes = 0
                if hold_minutes > max_hold_minutes:
                    hold_minutes = max_hold_minutes

                if decision == "VETO":
                    signal["action"] = "HOLD"
                    signal["reason"] = f"{signal.get('reason','')} [AI Veto] {str((ai_resp or {}).get('rationale',''))[:120]}"
                    if not use_cached and not should_skip_ai:
                        status("AI: VETO")
                else:
                    if not use_cached and not should_skip_ai:
                        status("AI: ALLOW")

                if decision not in {"ALLOW", "VETO"} and on_error == "veto":
                    signal["action"] = "HOLD"
                    signal["reason"] = f"{signal.get('reason','')} [AI Error Veto]"

            if signal['action'] != "HOLD":
                # Regime Vetoes
                if ai_enabled:
                    if regime == "BEARISH" and signal['action'] == "BUY":
                        signal['action'] = "HOLD"
                        signal['reason'] += " [AI Veto: Bearish regime]"
                    elif regime == "BULLISH" and signal['action'] == "SELL":
                        signal['action'] = "HOLD"
                        signal['reason'] += " [AI Veto: Bullish regime]"
                    elif regime == "VOLATILE":
                        signal['action'] = "HOLD"
                        signal['reason'] += " [AI Veto: Volatile regime]"

                # Re-check action after vetoes
                if signal['action'] != "HOLD":
                    entry_style = str(ai_overlay_state.get('entry_style', 'MIXED')).upper()
                    target_price = float(signal.get('entry', state['price']) or state['price'])
                    
                    if entry_style == "BUY_PULLBACKS" and signal['action'] == "BUY":
                        target_price = min(target_price, state['price'] * 0.999)
                    elif entry_style == "SELL_RALLIES" and signal['action'] == "SELL":
                        target_price = max(target_price, state['price'] * 1.001)

                    executor.place_limit_order(signal, symbol, target_price)

            curr_status = str(getattr(executor, 'last_status', '') or "")
            if curr_status != last_reported_status:
                status(f"Exec: {curr_status}")
                last_reported_status = curr_status

            # Render Dashboard
            if ticks % 5 == 0:
                open_orders_cache = executor.get_open_orders(symbol)
                
            ui_mode = _detect_ui_mode(str(ui_cfg.get("mode", "auto")))
            print_dashboard(
                ticks, symbol, regime, state, signal, executor, session_start,
                mtf_context=mtf_context, mtf_cfg=mtf_cfg,
                status_lines=list(status_buf)[-int(ui_cfg.get("status_lines", 3) or 3):],
                ui_cfg=ui_cfg, ai_overlay=ai_overlay_state, pivot_data=pivot_data,
                open_orders=open_orders_cache
            )

            time.sleep(tick_delay)

    except KeyboardInterrupt:
        logger.info("\nBot stopped by user (Ctrl+C). Initiating graceful shutdown...")
        print("\nBot stopped by user. Gracefully closing active positions and open orders...")
        executor.close_all_positions(symbol)
        print("Shutdown complete. All positions liquidated and orders cancelled.")
    finally:
        # Restore terminal if we hid cursor or used alt screen
        _show_cursor_ansi()
        _exit_alt_screen_ansi()

if __name__ == "__main__":
    run_hybrid_bot()
