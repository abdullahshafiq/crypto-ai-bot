import os
import sys
import time
import yaml
import logging
import shutil
import pandas as pd
from collections import deque
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv
import socket

_SINGLETON_SOCKET = None
def _enforce_single_instance(port=45678):
    global _SINGLETON_SOCKET
    try:
        _SINGLETON_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _SINGLETON_SOCKET.bind(("127.0.0.1", port))
    except OSError:
        print(f"ERROR: Another instance of the bot is already running. Please close it first.")
        sys.exit(1)


def _count_consecutive_losses(closed_trades) -> int:
    losses = 0
    for trade in reversed(list(closed_trades or [])):
        try:
            pnl = float(trade.get("pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl < 0:
            losses += 1
        else:
            break
    return losses

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from market_data import MarketData
from indicators import calculate_base_indicators, generate_quant_signal, build_mtf_timeframe_context, compute_advanced_pivots
from news_data import NewsData
from agents import HybridAIOrchestrator
import copy
from execution_factory import create_executor, resolve_data_market
from performance_gate import paper_gate_passed
from dashboard_server import DashboardRuntime, start_dashboard_server

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
        cfg = yaml.safe_load(f)
        
    try:
        import json
        overrides_path = "ui_state.json"
        if path != "config.yaml":
            base = os.path.splitext(os.path.basename(path))[0]
            overrides_path = f"{base}_state.json"
        if os.path.exists(overrides_path):
            with open(overrides_path, 'r') as f:
                ui_state = json.load(f)
            
            for path_key, val in ui_state.items():
                parts = path_key.split(".")
                cur = cfg
                for part in parts[:-1]:
                    if part not in cur:
                        cur[part] = {}
                    cur = cur[part]
                cur[parts[-1]] = val
    except Exception as e:
        logging.getLogger("main").warning(f"Failed to load overrides from {overrides_path}: {e}")
        
    return cfg

def _instance_port_for_config(cfg: dict) -> int:
    env_port = os.getenv("BOT_INSTANCE_PORT")
    if env_port:
        return int(env_port)
    exec_mode = str((cfg.get("execution", {}) or {}).get("mode", "live") or "live").strip().lower()
    return 45679 if exec_mode == "paper" else 45678

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
    if getattr(executor, "is_paper", False):
        val = executor.initial_balance
    else:
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
    def ln(label, value, label2="", value2="", label3="", value3="", width=frame_width):
        # We manually build the line to avoid layout shifting
        left = f" {YELLOW}{label}:{RESET} {value}"
        right = ""
        if label2:
            right = f" {YELLOW}{label2}:{RESET} {value2}"
        if label3:
            right = f"{right}  {YELLOW}{label3}:{RESET} {value3}" if right else f" {YELLOW}{label3}:{RESET} {value3}"
        
        # Estimate plain text length (labels are usually 5-10 chars, values vary)
        # To be safe, we just use the CL code to wipe the rest of the terminal line
        return f"{left}{'  ' + right if right else ''}{CL}"

    out = []
    def trim_text(text, max_len):
        text = str(text or "")
        return text if len(text) <= max_len else text[:max_len - 3] + "..."

    out.append(f"{CYAN}{BOLD}{'='*frame_width}{RESET}{CL}")
    exec_label = getattr(executor, "label", "EXEC")
    paused_indicator = f" {YELLOW}{BOLD}[PAUSED]{RESET}" if bool(ui_cfg.get("paused", False)) else ""
    bal_str = f"{GREEN}${val:,.2f}{RESET}" if val >= executor.initial_balance else f"{RED}${val:,.2f}{RESET}"
    out.append(f" {BOLD}🚀 {exec_label} | {symbol} | {regime_str}{paused_indicator}{RESET} | Bal: {bal_str} | {datetime.now().strftime('%H:%M:%S')} | {pnl_str}{CL}")
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

    # --- STRIKE ZONES (ENTRY TARGETS) ---
    m5 = mtf_context.get("5m", {}) if isinstance(mtf_context, dict) else {}
    m5_s, m5_r = nearest_levels(current_price, m5.get('support_levels', []), m5.get('resistance_levels', []))
    
    out.append(f" {BLUE}{BOLD}[ STRIKE ZONES ]{RESET}{CL}")
    bull_strike = f"{GREEN}${m5_r:.5f}{RESET}" if m5_r else f"{YELLOW}WAITING{RESET}"
    bear_strike = f"{RED}${m5_s:.5f}{RESET}" if m5_s else f"{YELLOW}WAITING{RESET}"
    out.append(f"  🎯 {GREEN}BULL STRIKE:{RESET} > {bull_strike}  🎯 {RED}BEAR STRIKE:{RESET} < {bear_strike}{CL}")

    out.append(f"{'-'*frame_width}{CL}")
    out.append(f" {BLUE}{BOLD}[ STRATEGY & AI ]{RESET}{CL}")
    out.append(ln("Action", f"{action_str} ({signal.get('confidence',0):.1%})", "Bias", f"{bias_str}", "Overlay", f"{overlay_bias_c}{overlay_bias}{RESET}"))
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
            sl_price = float(pos.get('sl', 0.0) or 0.0)
            support = float(pos.get('structure_support', 0.0) or 0.0)
            resistance = float(pos.get('structure_resistance', 0.0) or 0.0)
            ref_price = support if pos['side'] == 'LONG' else resistance
            ref_label = "S" if pos['side'] == 'LONG' else "R"
            ref_str = f"{ref_label}:${ref_price:.5f}" if ref_price > 0 else f"{ref_label}:--"
            trail_mode = "NATIVE" if bool(getattr(executor, 'use_native_trailing_stop', False)) else "LOCAL"
            trail_id = str(pos.get('native_trailing_order_id', '') or '')
            trail_id_str = f"#{trail_id[-6:]}" if trail_id else ""
            if pnl_pct >= 0.50:
                trail_stage = "T2"
            elif pnl_pct >= 0.25:
                trail_stage = "T1"
            elif pnl_pct >= 0.15:
                trail_stage = "BE+"
            else:
                trail_stage = "BASE"
            out.append(
                f"  {side_c}{pos['side']}{RESET} ${pos['amount']*pos['entry']:.0f} @ ${pos['entry']:.5f} | "
                f"{pnl_c}PnL:${val_pnl:+,.2f} ({pnl_pct:+.2f}%){RESET} | "
                f"SL:${sl_price:.5f} | {ref_str} | Trail:{trail_mode}{trail_id_str}/{trail_stage}{CL}"
            )

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
    try:
        sys.stdout.write('\n'.join(out) + tail_clear)
        sys.stdout.flush()
    except (OSError, ValueError):
        # Some Windows shells / IDE consoles expose a stdout handle that cannot be flushed reliably.
        # The dashboard should keep running even if the terminal renderer cannot repaint.
        return
    _LAST_FRAME_LINES = frame_lines

def _build_dashboard_snapshot(symbol, regime, state, signal, executor, session_start, status_lines, pivot_data, mtf_context, open_orders, latest_indicators, chart_bars, ai_overlay_state, cfg):
    current_price = float(state.get("price", 0.0) or 0.0) if isinstance(state, dict) else 0.0
    portfolio_value = float(getattr(executor, "initial_balance", 0.0) or 0.0)
    if not getattr(executor, "is_paper", False):
        try:
            portfolio_value = float(executor.get_portfolio_value(current_price))
        except Exception:
            pass
        pass
    pnl = portfolio_value - float(getattr(executor, "initial_balance", 0.0) or 0.0)
    pnl_pct = (pnl / float(executor.initial_balance) * 100.0) if float(getattr(executor, "initial_balance", 0.0) or 0.0) > 0 else 0.0
    positions = copy.deepcopy(getattr(executor, "active_positions", []) or [])
    pending_entry = copy.deepcopy(getattr(executor, "pending_entry", None))
    pending_exit = copy.deepcopy(getattr(executor, "pending_exit", None))
    closed_trades = list(getattr(executor, "closed_trades", []) or [])[-20:]
    realized_profit = 0.0
    realized_loss = 0.0
    realized_net = 0.0
    for trade in list(getattr(executor, "closed_trades", []) or []):
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        realized_net += pnl
        if pnl >= 0:
            realized_profit += pnl
        else:
            realized_loss += abs(pnl)
    stats_trades = int(getattr(executor, "stats_trades", len(getattr(executor, "closed_trades", []) or [])) or 0)
    stats_wins = int(getattr(executor, "stats_wins", 0) or 0)
    stats_losses = int(getattr(executor, "stats_losses", 0) or 0)
    win_rate = (stats_wins / stats_trades * 100.0) if stats_trades > 0 else 0.0
    # Calculate Unrealized PnL for active positions
    unrealized_pnl = 0.0
    total_active_cost = 0.0
    for pos in positions:
        pos_pnl = (current_price - float(pos['entry'])) if pos['side'] == 'LONG' else (float(pos['entry']) - current_price)
        unrealized_pnl += pos_pnl * float(pos['amount'])
        total_active_cost += float(pos['entry']) * float(pos['amount'])
    
    unrealized_pnl_pct = (unrealized_pnl / total_active_cost * 100.0) if total_active_cost > 0 else 0.0

    return {
        "ts": time.time(),
        "symbol": symbol,
        "regime": regime,
        "mode": getattr(executor, "label", "BOT"),
        "price": current_price,
        "spread_pct": float(state.get("spread_pct", 0.0) or 0.0) if isinstance(state, dict) else 0.0,
        "ret_30s": state.get("ret_30s") if isinstance(state, dict) else None,
        "balance": portfolio_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "pnl_stats": {
            "total_profit": realized_profit,
            "total_loss": realized_loss,
            "net_profit": realized_net,
            "wins": stats_wins,
            "losses": stats_losses,
            "trades": stats_trades,
            "win_rate": win_rate,
            "fees": float(getattr(executor, "stats_fees", 0.0) or 0.0),
        },
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "session_start": session_start,
        "uptime_sec": time.time() - session_start,
        "signal": copy.deepcopy(signal or {}),
        "positions": positions,
        "pending_entry": pending_entry,
        "pending_exit": pending_exit,
        "open_orders": copy.deepcopy(open_orders or []),
        "closed_trades": copy.deepcopy(closed_trades),
        "status_lines": list(status_lines or []),
        "pivot_data": copy.deepcopy(pivot_data or {}),
        "mtf_context": copy.deepcopy(mtf_context or {}),
        "latest_indicators": copy.deepcopy(latest_indicators or {}),
        "ai_overlay": copy.deepcopy(ai_overlay_state or {}),
        "chart": list(chart_bars or []),
        "config": {
            "execution": copy.deepcopy(cfg.get("execution", {}) or {}),
            "strategy": copy.deepcopy(cfg.get("strategy", {}) or {}),
            "mtf": copy.deepcopy(cfg.get("mtf", {}) or {}),
        }
    }

def _fallback_bootstrap_ohlcv(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Offline demo fallback used only when Binance market data is unreachable.
    Generates a minimal synthetic OHLCV history around a stable anchor so the
    paper executor can warm up and run locally.
    """
    limit = max(20, int(limit or 100))
    anchor = 9.10 if "AVAX" in str(symbol).upper() else 100.0
    if "BTC" in str(symbol).upper():
        anchor = 65000.0
    now = pd.Timestamp.utcnow().floor("min")
    try:
        if str(timeframe).endswith("m"):
            delta = pd.Timedelta(minutes=int(str(timeframe)[:-1] or 1))
        elif str(timeframe).endswith("h"):
            delta = pd.Timedelta(hours=int(str(timeframe)[:-1] or 1))
        else:
            delta = pd.Timedelta(minutes=1)
    except Exception:
        delta = pd.Timedelta(minutes=1)
    rows = []
    price = float(anchor)
    for i in range(limit):
        ts = now - delta * (limit - i)
        drift = ((i % 7) - 3) * (anchor * 0.0002)
        open_p = price
        close_p = max(0.0001, price + drift)
        high_p = max(open_p, close_p) * 1.0008
        low_p = min(open_p, close_p) * 0.9992
        vol = 1000.0 + (i % 10) * 25.0
        rows.append({
            "timestamp": ts,
            "open": float(open_p),
            "high": float(high_p),
            "low": float(low_p),
            "close": float(close_p),
            "volume": float(vol),
        })
        price = close_p
    return pd.DataFrame(rows)


_NETWORK_FAIL_UNTIL = 0.0
_NETWORK_COOLDOWN_SECONDS = 60.0

def _runtime_fetch_ohlcv(market, symbol: str, timeframe: str, limit: int, *, paper_mode: bool, logger=None) -> pd.DataFrame:
    """
    Fetch OHLCV for runtime use. In paper mode, fall back to a synthetic bootstrap
    when Binance market data is unreachable so the demo can still run locally.
    """
    global _NETWORK_FAIL_UNTIL
    now = time.time()
    if now < _NETWORK_FAIL_UNTIL:
        if paper_mode:
            return _fallback_bootstrap_ohlcv(symbol, timeframe, limit=limit)
        return pd.DataFrame()

    try:
        df = market.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if df is not None and not df.empty:
            _NETWORK_FAIL_UNTIL = 0.0
            return df
    except Exception as e:
        if logger:
            logger.debug(f"Runtime OHLCV fetch failed for {symbol} {timeframe}: {e}")

    _NETWORK_FAIL_UNTIL = now + _NETWORK_COOLDOWN_SECONDS
    if paper_mode:
        if logger:
            logger.warning(f"Paper runtime falling back to synthetic OHLCV for {symbol} {timeframe}.")
        return _fallback_bootstrap_ohlcv(symbol, timeframe, limit=limit)
    return pd.DataFrame()


def _reapply_runtime_executor_config(executor, cfg):
    exec_cfg = cfg.get("execution", {}) or {}
    executor.max_open_positions = cfg["risk"].get("max_open_positions", 1)
    executor.daily_loss_cap_pct = cfg["risk"].get("daily_loss_cap")
    executor.min_balance_floor = float(cfg["risk"].get("min_balance_floor", 90.0))
    leverage_cfg = cfg.get("leverage", {}) or {}
    executor.dynamic_leverage_enabled = bool(leverage_cfg.get("enabled", False))
    executor.leverage_min = float(leverage_cfg.get("min_leverage", 1.0))
    executor.leverage_max = float(leverage_cfg.get("max_leverage", 4.0))
    executor.leverage_use_score = bool(leverage_cfg.get("use_score_multiplier", False))
    executor.leverage_score_weight = float(leverage_cfg.get("score_weight", 0.3))
    executor.atr_volatility_scaling = bool(leverage_cfg.get("atr_volatility_scaling", False))
    executor.atr_reference_pct = float(leverage_cfg.get("atr_reference_pct", 0.02))
    executor.atr_min_multiplier = float(leverage_cfg.get("atr_min_multiplier", 0.3))
    conf_levels = leverage_cfg.get("confidence_levels", {})
    executor.leverage_confidence_levels = {float(k): float(v) for k, v in conf_levels.items()}
    executor.dca_enabled = bool(exec_cfg.get("dca_enabled", False))
    executor.dca_max_steps = int(exec_cfg.get("dca_max_steps", 0))
    executor.dca_distance_pct = float(exec_cfg.get("dca_distance_pct", 0.01))

def run_hybrid_bot():
    load_dotenv()

    config_path = os.environ.get("BOT_CONFIG", "config.yaml")
    cfg = load_config(config_path)
    _enforce_single_instance(_instance_port_for_config(cfg))
    _enable_windows_vt_mode()
    setup_logging(cfg.get("logging", {}) or {})
    logger = logging.getLogger("main")
    symbol = cfg['symbol']
    mtf_cfg = cfg.get("mtf", {}) or {}
    ai_trade_cfg = cfg.get("ai_trade", {}) or {}
    ai_overlay_cfg = cfg.get("ai_overlay", {}) or cfg.get("ai", {}) or {}
    if "enabled" not in ai_overlay_cfg and "overlay_enabled" in ai_overlay_cfg:
        ai_overlay_cfg = dict(ai_overlay_cfg)
        ai_overlay_cfg["enabled"] = ai_overlay_cfg.get("overlay_enabled")
    if "refresh_seconds" not in ai_overlay_cfg and "overlay_refresh_seconds" in ai_overlay_cfg:
        ai_overlay_cfg = dict(ai_overlay_cfg)
        ai_overlay_cfg["refresh_seconds"] = ai_overlay_cfg.get("overlay_refresh_seconds")
    ui_cfg = cfg.setdefault("ui", {})
    mem_cfg = cfg.get("memory", {}) or {}
    auto_learning_cfg = cfg.get("auto_learning", {}) or {}

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    print("Booting up Hybrid Crypto AI Bot...")

    exec_cfg = cfg.get("execution", {}) or {}
    leverage = int(exec_cfg.get("leverage", 5))
    requested_exec_mode = str(exec_cfg.get("mode", os.getenv("EXECUTION_MODE", "live"))).strip().lower()

    strategy_config = {
        'max_spread': cfg['strategy']['max_spread'],
        'min_conf': cfg['strategy']['min_conf'],
        'fixed_trade_usdt': cfg['strategy'].get('fixed_trade_usdt', 100),
        'tp_pct': cfg['strategy']['tp_pct'],
        'sl_pct': cfg['strategy']['sl_pct'],
        'max_structural_sl_pct': cfg['strategy'].get('max_structural_sl_pct', 0.0030),
        'min_reward_risk': cfg['strategy'].get('min_reward_risk', 0.90),
        'max_ret_30s': cfg['strategy'].get('max_ret_30s', 0.0050),
        'max_ret_5s': cfg['strategy'].get('max_ret_5s', 0.0025),
        'block_on_volume_spike': cfg['strategy'].get('block_on_volume_spike', False),
        'vol_filter_atr_pct': cfg['strategy'].get('vol_filter_atr_pct', 0.0005),
        'vol_filter_atr_max_pct': cfg['strategy'].get('vol_filter_atr_max_pct', 0.08),
        # Pass timeframe so indicators.py can compute correct candle lookback for range veto
        'timeframe': cfg.get('timeframe', '1m'),
    }
    fixed_trade_usdt = float(strategy_config.get('fixed_trade_usdt', 0.0) or 0.0)

    indicator_refresh_interval = cfg['intervals']['indicator_refresh']
    regime_refresh_interval = cfg['intervals']['regime_refresh']
    tick_delay = cfg['intervals']['tick_delay_seconds']
    ai_enabled = cfg['ai']['enabled']
    ai_model = cfg['ai']['model']
    
    data_market = resolve_data_market(cfg)
    market = MarketData(market=data_market)
    news = NewsData()
    ai_orch = HybridAIOrchestrator(model=ai_model)
    
    print("Fetching historical data to warm up indicators (EMAs/RSI/VWAP)...")
    paper_bootstrapped_synthetic = False
    paper_mode = requested_exec_mode == "paper"
    hist_df = _runtime_fetch_ohlcv(market, symbol, cfg['timeframe'], 100, paper_mode=paper_mode, logger=logger)
    if hist_df.empty:
        # Fallback: try the other market type once (spot <-> usdm) for convenience.
        alt_market = "spot" if getattr(market, "market", "usdm") == "usdm" else "usdm"
        alt = MarketData(market=alt_market)
        alt_df = _runtime_fetch_ohlcv(alt, symbol, cfg['timeframe'], 100, paper_mode=paper_mode, logger=logger)
        if not alt_df.empty:
            market = alt
            hist_df = alt_df
            print(f"Note: switched market data source to `{alt_market}` for symbol {symbol}.")
        else:
            if paper_mode:
                hist_df = _fallback_bootstrap_ohlcv(symbol, cfg['timeframe'], limit=100)
                paper_bootstrapped_synthetic = True
                print(f"{YELLOW}{BOLD}Paper demo bootstrapping synthetic candles for {symbol} ({cfg['timeframe']}).{RESET}")
                logger.warning("Using synthetic OHLCV bootstrap for paper demo startup.")
            else:
                print(f"CRITICAL ERROR: Failed to fetch initial OHLCV data for {symbol} ({cfg['timeframe']}).")
                print("Tip: check symbol spelling and whether it exists on Futures (usdm) vs Spot.")
                print("Tip: try `symbol: \"BTC/USDT\"` to verify connectivity.")
                logger.critical("Failed to fetch initial OHLCV data. Exiting.")
                return

    df_indicators = calculate_base_indicators(hist_df)
    latest_indicators = (df_indicators.iloc[-2] if len(df_indicators) > 1 else df_indicators.iloc[-1]).to_dict()
    bootstrap_price = float(hist_df.iloc[-1]['close'])
    # Initialize chart_bars with history for dashboard
    chart_bars = deque(maxlen=int(cfg.get("dashboard", {}).get("candle_limit", 240)) or 240)
    for _, row in hist_df.iterrows():
        chart_bars.append({
            "time": int(row["timestamp"].timestamp() * 1000),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    if requested_exec_mode == "live":
        exec_market = str(exec_cfg.get("market", "usdm") or "usdm").strip().lower()
        gate_log_file = "trade_log_spot.csv" if exec_market == "spot" else "trade_log_futures.csv"
        passed, metrics = paper_gate_passed(
            log_file=gate_log_file,
            min_trades=int(exec_cfg.get("paper_gate_min_trades", 100)),
            min_profit_factor=float(exec_cfg.get("paper_gate_min_profit_factor", 1.2)),
            max_drawdown=float(exec_cfg.get("paper_gate_max_drawdown", 0.20)),
        )
        if not passed:
            logger.warning("Live mode paper gate failed, but live-only runtime will continue: %s", metrics)
            print(f"{YELLOW}{BOLD}Paper safety gate failed, but live-only runtime continues.{RESET}")

    executor = create_executor(
        cfg=cfg,
        api_key=api_key,
        api_secret=api_secret,
        bootstrap_price=bootstrap_price,
        fixed_trade_usdt=fixed_trade_usdt,
    )
    executor.symbol = symbol

    if cfg.get("execution", {}).get("mode", "").lower() == "paper":
        executor.label = "PAPER"

    executor.max_open_positions = cfg['risk'].get('max_open_positions', 1)
    executor.daily_loss_cap_pct = cfg['risk'].get('daily_loss_cap')
    executor.min_balance_floor = float(cfg['risk'].get('min_balance_floor', 90.0))

    auto_learning_enabled = bool(auto_learning_cfg.get("enabled", False))
    auto_learning_min_trades = max(50, int(auto_learning_cfg.get("min_completed_trades", 50)))
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
    status_buf = deque(maxlen=80)
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
    last_reported_signal = ""
    last_learning_closed_trades = int(getattr(executor, "stats_trades", 0) or 0)
    loss_tilt_pause_until = 0.0
    loss_tilt_last_count = 0
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

    signal_history = deque(maxlen=5) # 5-second smoothing buffer
    # chart_bars initialized above
    last_chart_tf = cfg.get("timeframe", "5m")

    dashboard_cfg = cfg.get("dashboard", {}) or {}
    dashboard_runtime = None
    if dashboard_cfg.get("enabled", True):
        config_file = config_path
        overrides_file = "ui_state.json" if config_file == "config.yaml" else f"{os.path.splitext(os.path.basename(config_file))[0]}_state.json"
        dashboard_runtime = DashboardRuntime(cfg, overrides_path=overrides_file)
        dashboard_host = dashboard_cfg.get("host", "127.0.0.1")
        dashboard_port = int(dashboard_cfg.get("port", 8080))
        dashboard_runtime.ensure_running(dashboard_host, dashboard_port)

    def status(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        status_buf.append(f"{ts} {msg}")

    try:
        while True:
            ticks += 1

            # Timeframe Switch Logic
            ui_tf = cfg.get("ui", {}).get("chart_tf", cfg.get("timeframe", "5m"))
            if ui_tf != last_chart_tf:
                # print(f"[DEBUG] TF Switch Detected: {last_chart_tf} -> {ui_tf}")
                status(f"Switching Chart to {ui_tf}")
                try:
                    new_hist = _runtime_fetch_ohlcv(market, symbol, ui_tf, 100, paper_mode=paper_mode, logger=logger)
                    if not new_hist.empty:
                        chart_bars.clear()
                        for _, row in new_hist.iterrows():
                            chart_bars.append({
                                "time": int(row["timestamp"].timestamp() * 1000),
                                "open": float(row["open"]), "high": float(row["high"]),
                                "low": float(row["low"]), "close": float(row["close"]),
                                "volume": float(row["volume"]),
                            })
                        last_chart_tf = ui_tf
                        status(f"Chart TF: {ui_tf} Loaded")
                    else:
                        status(f"Chart TF {ui_tf}: fetch returned empty data, retrying next cycle")
                except Exception as e:
                    logger.error(f"Failed to switch chart TF: {e}")

            if ticks == 1 or ticks % indicator_refresh_interval == 0:
                logger.info("Refreshing macro indicators...")
                status("Refreshing indicators/MTF")
                # Main execution OHLCV (always use bot's execution timeframe)
                new_df = _runtime_fetch_ohlcv(market, symbol, cfg['timeframe'], 100, paper_mode=paper_mode, logger=logger)
                macro_df = _runtime_fetch_ohlcv(market, symbol, cfg.get('macro_timeframe', '1h'), 100, paper_mode=paper_mode, logger=logger)

                # Chart-specific OHLCV update (use dashboard's selected timeframe)
                try:
                    chart_update_df = _runtime_fetch_ohlcv(market, symbol, ui_tf, 5, paper_mode=paper_mode, logger=logger)
                    if not chart_update_df.empty:
                        seen_times = {b["time"] for b in chart_bars}
                        for _, row in chart_update_df.iterrows():
                            new_bar = {
                                "time": int(row["timestamp"].timestamp() * 1000), 
                                "open": float(row["open"]), "high": float(row["high"]), 
                                "low": float(row["low"]), "close": float(row["close"]), 
                                "volume": float(row["volume"])
                            }
                            if new_bar["time"] not in seen_times:
                                chart_bars.append(new_bar)
                                seen_times.add(new_bar["time"])
                except Exception as e:
                    logger.warning(f"Chart periodic update failed: {e}")
                    
                if mtf_cfg.get("enabled", False):
                    for tf in mtf_cfg.get("timeframes", ["15m", "3h", "4h"]):
                        htf_df = _runtime_fetch_ohlcv(market, symbol, tf, 200, paper_mode=paper_mode, logger=logger)
                        if htf_df.empty:
                            status(f"MTF {tf}: fetch failed")
                            continue
                        htf_ind = calculate_base_indicators(htf_df)
                        if len(htf_ind) > 1:
                            htf_ind = htf_ind.iloc[:-1]
                        ctx = build_mtf_timeframe_context(htf_ind)
                        ctx["computed_at"] = time.time()
                        mtf_context[str(tf)] = ctx
                    status("MTF context updated")

                if not new_df.empty:
                    df_indicators = calculate_base_indicators(new_df)
                    latest_indicators = (df_indicators.iloc[-2] if len(df_indicators) > 1 else df_indicators.iloc[-1]).to_dict()

                # Refresh Advanced Pivot Points from daily OHLCV (every 15 min)
                if time.time() - last_pivot_refresh_ts >= 900 or not pivot_data:
                    try:
                        daily_df = _runtime_fetch_ohlcv(market, symbol, '1d', 5, paper_mode=paper_mode, logger=logger)
                        if not daily_df.empty and len(daily_df) >= 2:
                            pivot_data = compute_advanced_pivots(daily_df)
                            last_pivot_refresh_ts = time.time()
                            classic_pp = pivot_data.get('classic', {})
                            if classic_pp:
                                status(f"Pivots: PP={classic_pp['pp']:.5f} S1={classic_pp['s1']:.5f} R1={classic_pp['r1']:.5f}")
                    except Exception as e:
                        logger.warning(f"Failed to compute pivots: {e}")

            is_ai_enabled = bool(cfg.get("ai", {}).get("enabled", False))
            ai_orch.enabled = is_ai_enabled
            if is_ai_enabled and (ticks == 1 or ticks % regime_refresh_interval == 0):
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

            # Sync dynamic overrides/config
            target_mode = str(cfg.get("execution", {}).get("mode", getattr(executor, "label", "paper"))).lower()
            current_label = str(getattr(executor, "label", "paper") or "paper").upper()
            is_live_executor = "LIVE" in current_label and "PAPER" not in current_label
            is_paper_executor = "PAPER" in current_label
            if target_mode == "live" and not is_live_executor:
                logger.info("Switching to LIVE execution mode as requested.")
                try:
                    executor.close_all_positions(symbol)
                except Exception:
                    pass
                cfg.setdefault("execution", {})["mode"] = "live"
                executor = create_executor(
                    cfg=cfg,
                    api_key=os.getenv("BINANCE_API_KEY"),
                    api_secret=os.getenv("BINANCE_SECRET"),
                    bootstrap_price=float(state.get("price", 0.0) or 0.0),
                    fixed_trade_usdt=fixed_trade_usdt,
                )
                executor.symbol = symbol
                executor.label = "LIVE"
                _reapply_runtime_executor_config(executor, cfg)
            elif target_mode == "paper" and not is_paper_executor:
                logger.info("Switching to PAPER execution mode as requested.")
                try:
                    executor.close_all_positions(symbol)
                except Exception:
                    pass
                cfg.setdefault("execution", {})["mode"] = "paper"
                executor = create_executor(
                    cfg=cfg,
                    api_key=os.getenv("BINANCE_API_KEY"),
                    api_secret=os.getenv("BINANCE_SECRET"),
                    bootstrap_price=float(state.get("price", 0.0) or 0.0),
                    fixed_trade_usdt=fixed_trade_usdt,
                )
                executor.symbol = symbol
                executor.label = "PAPER"
                _reapply_runtime_executor_config(executor, cfg)

            is_paused = bool(cfg.get("execution", {}).get("paused", False))
            if is_paused:
                # print(f"[DEBUG] Bot is currently PAUSED")
                pass
            executor.paused = is_paused
            if is_paused:
                if getattr(executor, "active_positions", None) or getattr(executor, "pending_entry", None) or getattr(executor, "pending_exit", None):
                    logger.info("Bot paused - immediately closing all positions and orders.")
                    try:
                        executor.close_all_positions(symbol)
                    except Exception as e:
                        logger.error(f"Error closing positions on pause: {e}")

            # Update executor with current ATR for volatility-based leverage
            atr_pct = latest_indicators.get('atr_pct')
            if atr_pct is not None and pd.notna(atr_pct):
                executor._current_atr_pct = float(atr_pct)

            # Pass PSAR to executor for dynamic trailing stop logic
            psar = latest_indicators.get('psar')
            if psar is not None and pd.notna(psar):
                executor._current_psar = float(psar)

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

            runtime_strategy_config = dict(strategy_config)
            runtime_strategy_config["min_conf"] = max(float(runtime_strategy_config.get("min_conf", 0.15) or 0.15), 0.15)
            runtime_strategy_config["entry_min_confidence_hard"] = max(
                float(runtime_strategy_config.get("entry_min_confidence_hard", 0.20) or 0.20),
                0.20,
            )
            closed_snapshot = list(getattr(executor, "closed_trades", []) or [])[-10:]
            consec_losses = _count_consecutive_losses(closed_snapshot)
            tilt_min_losses = max(1, int(strategy_config.get("loss_tilt_min_losses", 3) or 3))
            tilt_pause_losses = max(tilt_min_losses + 1, int(strategy_config.get("loss_tilt_pause_losses", 5) or 5))
            tilt_pause_minutes = max(1, int(strategy_config.get("loss_tilt_pause_minutes", 15) or 15))
            if consec_losses >= tilt_min_losses:
                runtime_strategy_config["min_conf"] = max(float(runtime_strategy_config.get("min_conf", 0.15) or 0.15), 0.50)
                runtime_strategy_config["entry_min_confidence_hard"] = max(
                    float(runtime_strategy_config.get("entry_min_confidence_hard", 0.20) or 0.20),
                    0.50,
                )
                runtime_strategy_config["midrange_min_score"] = max(
                    float(runtime_strategy_config.get("midrange_min_score", 0.28) or 0.28),
                    0.50,
                )
                runtime_strategy_config["session_block_min_score"] = max(
                    float(runtime_strategy_config.get("session_block_min_score", 0.35) or 0.35),
                    0.50,
                )
            if consec_losses >= tilt_pause_losses and consec_losses > loss_tilt_last_count:
                loss_tilt_pause_until = max(loss_tilt_pause_until, time.time() + float(tilt_pause_minutes * 60))
            loss_tilt_last_count = consec_losses

            signal_df = df_indicators.iloc[:-1] if (df_indicators is not None and len(df_indicators) > 1) else df_indicators
            
            signal = generate_quant_signal(
                    state,
                    latest_indicators,
                    runtime_strategy_config,
                    signal_df,
                    latest_macro,
                    mtf_context=mtf_context,
                    mtf_config=mtf_cfg,
                    pivot_data=pivot_data,
                )
            
            if time.time() < loss_tilt_pause_until and signal.get("action") in {"BUY", "SELL"}:
                signal["action"] = "HOLD"
                signal["hold_reason"] = f"Consecutive loss tilt: {tilt_pause_minutes}m entry pause"
                signal["reason"] = f"{signal.get('reason','')} LOSS_TILT_PAUSE"
            
            # SIGNAL SMOOTHING: Simplified for faster scalp entry.
            raw_score = float(signal.get('score', 0.0) or 0.0)
            signal_history.append(raw_score)
            avg_score = sum(signal_history) / len(signal_history)
            
            # Apply the score and confidence
            signal['score'] = raw_score # Use raw tick for speed
            signal['confidence'] = abs(max(-1.0, min(1.0, raw_score)))
            
            # Only hold if the RAW confidence is truly under the floor
            if signal['confidence'] < float(runtime_strategy_config.get('min_conf', 0.05)):
                signal['action'] = "HOLD"
                signal['hold_reason'] = "Weak confidence (<5%)"

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
                    signal["reason"] = f"{signal.get('reason','')} [AI Overlay Soft Bias] {overlay_note}"
                elif overlay_bias == "SHORT_ONLY" and signal.get("action") == "BUY":
                    signal["reason"] = f"{signal.get('reason','')} [AI Overlay Soft Bias] {overlay_note}"

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
                scalp_friendly = (
                    signal.get("action") in {"BUY", "SELL"}
                    and float(signal.get("confidence", 0.0) or 0.0) >= float(runtime_strategy_config.get("min_conf", 0.05))
                    and not bool(ai_overlay_state.get("avoid_new_entries", False))
                    and str(ai_overlay_state.get("risk_mode", "NORMAL") or "NORMAL").upper() not in {"HIGH", "EXTREME"}
                )

                if decision == "VETO":
                    veto_note = str((ai_resp or {}).get("rationale", "") or "")[:120]
                    if scalp_friendly:
                        signal["reason"] = f"{signal.get('reason','')} [AI Soft Veto Ignored] {veto_note}"
                        if not use_cached and not should_skip_ai:
                            status("AI: SOFT ALLOW")
                    else:
                        signal["action"] = "HOLD"
                        signal["reason"] = f"{signal.get('reason','')} [AI Veto] {veto_note}"
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
                if is_ai_enabled:
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
            if not is_paused and signal['action'] != "HOLD":
                    entry_style = str(ai_overlay_state.get('entry_style', 'MIXED')).upper()
                    target_price = float(signal.get('entry', state['price']) or state['price'])

                    if entry_style == "BUY_PULLBACKS" and signal['action'] == "BUY":
                        target_price = min(target_price, state['price'] * 0.999)
                    elif entry_style == "SELL_RALLIES" and signal['action'] == "SELL":
                        target_price = max(target_price, state['price'] * 1.001)

                    # Structural Take Profit Interception
                    if signal['action'] == "BUY" and signal.get("structure_resistance"):
                        signal["tp_target"] = float(signal["structure_resistance"]) * 0.999
                    elif signal['action'] == "SELL" and signal.get("structure_support"):
                        signal["tp_target"] = float(signal["structure_support"]) * 1.001

                    executor.place_limit_order(signal, symbol, target_price)

            curr_status = str(getattr(executor, 'last_status', '') or "")
            if curr_status != last_reported_status:
                status(f"Exec: {curr_status}")
                last_reported_status = curr_status

            # Render Dashboard
            if ticks % 5 == 0:
                open_orders_cache = executor.get_open_orders(symbol)
            ui_mode = _detect_ui_mode(str(ui_cfg.get("mode", "auto")))
            ui_cfg["paused"] = is_paused # Pass pause state to UI
            print_dashboard(
                ticks, symbol, regime, state, signal, executor, session_start,
                mtf_context=mtf_context, mtf_cfg=mtf_cfg,
                status_lines=list(status_buf)[-int(ui_cfg.get("status_lines", 3) or 3):],
                ui_cfg=ui_cfg, ai_overlay=ai_overlay_state, pivot_data=pivot_data,
                open_orders=open_orders_cache
            )

            if dashboard_runtime is not None:
                if ticks == 1 or ticks % 20 == 0:
                    dashboard_runtime.ensure_running(dashboard_host, dashboard_port)
                dashboard_runtime.update_state(
                    _build_dashboard_snapshot(
                        symbol, regime, state, signal, executor, session_start,
                        list(status_buf)[-int(ui_cfg.get("status_lines", 3) or 3):],
                        pivot_data, mtf_context, open_orders_cache, latest_indicators,
                        chart_bars, ai_overlay_state, cfg
                    )
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
