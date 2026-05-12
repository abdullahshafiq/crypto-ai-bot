import os
import sys
import time
import shutil
from datetime import datetime

from . import windows_vt as _vt
from .windows_vt import (
    _detect_ui_mode, _enable_windows_vt_mode, _enter_alt_screen_ansi,
    _windows_cursor_home, _hide_cursor_ansi, _show_cursor_ansi, _exit_alt_screen_ansi,
)
from .windows_vt import CYAN, MAGENTA, YELLOW, BLUE, GREEN, RED, BOLD, RESET, CL


def print_dashboard(ticks, symbol, regime, state, signal, executor, session_start, mtf_context=None, mtf_cfg=None, status_lines=None, ui_cfg=None, ai_overlay=None, pivot_data=None, open_orders=None, freshness=None):
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
    out.append(ln("Intent", trim_text(signal.get('intent', 'Waiting for alignment'), frame_width - 10)))
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
            eff_lev = float(pos.get('effective_leverage', getattr(executor, 'leverage', 1.0)) or getattr(executor, 'leverage', 1.0))
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
                f"Lev:{eff_lev:.1f}x | SL:${sl_price:.5f} | {ref_str} | Trail:{trail_mode}{trail_id_str}/{trail_stage}{CL}"
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

    if isinstance(freshness, dict):
        parts = []
        lp = freshness.get("live_price_age")
        if lp is not None and lp >= 0:
            parts.append(f"Live:{lp:.1f}s")
        sc = freshness.get("signal_candle_age")
        if sc is not None and sc >= 0:
            parts.append(f"Signal:{sc:.0f}s")
        mtf = freshness.get("mtf_ages", {})
        if mtf:
            mtf_parts = ",".join(f"{tf}:{v:.0f}s" for tf, v in sorted(mtf.items()))
            parts.append(f"MTF:{mtf_parts}")
        if parts:
            out.append(f" {YELLOW}Fresh:{RESET} {'  '.join(parts)}{CL}")

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
    if ui_mode == "win" and _vt._LAST_FRAME_LINES > frame_lines:
        for _ in range(_vt._LAST_FRAME_LINES - frame_lines):
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
    _vt._LAST_FRAME_LINES = frame_lines
