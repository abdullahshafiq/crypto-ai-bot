import os
import sys
import time
import logging
import copy
import re
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

from market import MarketData
from indicators import (
    calculate_base_indicators,
    generate_quant_signal,
    build_mtf_timeframe_context,
    compute_advanced_pivots,
)
from market import NewsData
from ai import HybridAIOrchestrator
from execution import create_executor, resolve_data_market
from safety import paper_gate_passed
from dashboard import DashboardRuntime, start_dashboard_server

from config.loader import load_config, _instance_port_for_config
from core.singles import _enforce_single_instance, _count_consecutive_losses
from core.snapshot import _build_dashboard_snapshot
from core.synthetic_data import _fallback_bootstrap_ohlcv
from core.startup import (
    _runtime_fetch_ohlcv,
    _startup_symbol_candidates,
    _reapply_runtime_executor_config,
    setup_logging,
)
from core.session import create_session_paths, write_session_manifest
from core.signal_gates import (
    compute_loss_tilt_override,
    apply_loss_tilt_pause,
    apply_scalp_hold_guard,
    apply_confidence_floor,
    apply_loss_tilt_hard_gate,
    apply_reward_risk_gate,
    format_gate_trace,
    apply_aggressive_scalp_gate,
)
from core.ai_gates import (
    apply_ai_overlay,
    apply_ai_trade_gate,
    apply_regime_veto,
    dispatch_entry,
)
from ui.terminal import print_dashboard, YELLOW, BOLD, RESET
from ui.windows_vt import (
    _enable_windows_vt_mode,
    _show_cursor_ansi,
    _exit_alt_screen_ansi,
    _detect_ui_mode,
)


def _timeframe_to_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").strip()
    match = re.fullmatch(r"(\d+)\s*([smhdwSMHDW]?)", tf)
    if not match:
        return 0
    value = int(match.group(1))
    unit = (match.group(2) or "m").lower()
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }
    return value * multipliers.get(unit, 0)


def _closed_chart_candles(df: pd.DataFrame, timeframe: str, include_live: bool = False) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    tf_seconds = _timeframe_to_seconds(timeframe)
    if tf_seconds <= 0:
        return df.reset_index(drop=True)
    closed = df.copy()
    now_utc = pd.Timestamp.utcnow()
    last_close = closed["timestamp"].iloc[-1] + pd.Timedelta(seconds=tf_seconds)
    if not include_live and last_close > now_utc:
        closed = closed.iloc[:-1]
    return closed.reset_index(drop=True)


def _chart_bar_payload(row: pd.Series) -> dict:
    """Build a dashboard candle payload with overlay indicator values."""
    def _num(key: str) -> float:
        val = row.get(key, None)
        try:
            return float(val) if val is not None and pd.notna(val) else 0.0
        except Exception:
            return 0.0

    payload = {
        "time": int(row["timestamp"].timestamp() * 1000),
        "open": _num("open"),
        "high": _num("high"),
        "low": _num("low"),
        "close": _num("close"),
        "volume": _num("volume"),
        "ema_9": _num("ema_9"),
        "ema_21": _num("ema_21"),
        "macd": _num("macd"),
        "macd_signal": _num("macd_signal"),
        "macd_diff": _num("macd_diff"),
        "psar": _num("psar"),
        "psar_streak": _num("psar_streak"),
    }
    return payload


def run_hybrid_bot():
    load_dotenv()

    config_path = os.environ.get("BOT_CONFIG", "config.yaml")
    cfg = load_config(config_path)
    _enforce_single_instance(_instance_port_for_config(cfg))
    _enable_windows_vt_mode()
    exec_cfg = cfg.get("execution", {}) or {}
    leverage = int(exec_cfg.get("leverage", 5))
    requested_exec_mode = str(exec_cfg.get("mode", os.getenv("EXECUTION_MODE", "live"))).strip().lower()
    logging_cfg = cfg.get("logging", {}) or {}
    session_root = str(logging_cfg.get("session_root", "sessions") or "sessions")
    session_paths = create_session_paths(
        session_root=session_root,
        mode=requested_exec_mode,
        symbol=cfg.get("symbol", "unknown"),
        timeframe=cfg.get("timeframe", "unknown"),
        trade_log_name="trade_log.csv",
    )
    logging_cfg = dict(logging_cfg)
    logging_cfg["log_file"] = session_paths.log_file
    cfg["logging"] = logging_cfg
    exec_cfg["trade_log_file"] = session_paths.trade_log_file
    cfg["execution"] = exec_cfg
    setup_logging(logging_cfg, log_file=session_paths.log_file)
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
    ui_cfg.setdefault("chart_tf", cfg.get("timeframe", "5m"))
    mem_cfg = cfg.get("memory", {}) or {}
    auto_learning_cfg = cfg.get("auto_learning", {}) or {}

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    print("Booting up Hybrid Crypto AI Bot...")

    strategy_config = dict(cfg.get("strategy", {}) or {})
    # Keep the full strategy block available to indicators.py so paper/live configs
    # can tune gates such as range veto, midrange score, and session filters.
    strategy_config.setdefault("timeframe", cfg.get("timeframe", "1m"))
    strategy_config.setdefault("max_spread", 0.0005)
    strategy_config.setdefault("min_conf", 0.15)
    strategy_config.setdefault("fixed_trade_usdt", 100)
    strategy_config.setdefault("tp_pct", 0.0035)
    strategy_config.setdefault("sl_pct", 0.0025)
    strategy_config.setdefault("max_structural_sl_pct", 0.0030)
    strategy_config.setdefault("min_reward_risk", 0.90)
    strategy_config.setdefault("max_ret_30s", 0.0050)
    strategy_config.setdefault("max_ret_5s", 0.0025)
    strategy_config.setdefault("block_on_volume_spike", False)
    strategy_config.setdefault("vol_filter_atr_pct", 0.0005)
    strategy_config.setdefault("vol_filter_atr_max_pct", 0.08)
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
    resolved_symbol = symbol
    hist_df = pd.DataFrame()
    for candidate_symbol in _startup_symbol_candidates(symbol):
        hist_df = market.fetch_ohlcv(
            candidate_symbol,
            timeframe=cfg['timeframe'],
            limit=200,
        )
        if hist_df is not None and not hist_df.empty:
            resolved_symbol = candidate_symbol
            if candidate_symbol != symbol:
                msg = f"{YELLOW}{BOLD}Symbol fallback: {symbol} -> {candidate_symbol}{RESET}"
                print(msg)
                logger.warning(
                    "Startup symbol fallback applied: %s -> %s",
                    symbol,
                    candidate_symbol,
                )
            break
        logger.warning(
            "Startup OHLCV fetch failed for %s (%s); trying fallback if available.",
            candidate_symbol,
            cfg['timeframe'],
        )

    if hist_df.empty:
        if paper_mode:
            hist_df = _fallback_bootstrap_ohlcv(
                resolved_symbol,
                cfg['timeframe'],
                limit=200,
            )
            paper_bootstrapped_synthetic = True
            msg = (
                f"{YELLOW}{BOLD}Paper demo bootstrapping synthetic candles "
                f"for {resolved_symbol} ({cfg['timeframe']}).{RESET}"
            )
            print(msg)
            logger.warning("Using synthetic OHLCV bootstrap for paper demo startup.")
        else:
            timeframe = cfg['timeframe']
            print(f"CRITICAL ERROR: Failed to fetch initial OHLCV data for {symbol} ({timeframe}).")
            print("Tip: this bot is futures-only and uses Binance USDⓈ-M data.")
            print("Tip: use a supported Binance USDⓈ-M futures symbol like `AVAX/USDT:USDT`.")
            logger.critical("Failed to fetch initial OHLCV data. Exiting.")
            return
    symbol = resolved_symbol
    cfg["symbol"] = resolved_symbol
    strategy_config["symbol"] = resolved_symbol
    write_session_manifest(
        session_paths,
        {
            "config_path": config_path,
            "execution_mode": requested_exec_mode,
            "symbol": symbol,
            "timeframe": cfg["timeframe"],
            "resolved_symbol": resolved_symbol,
            "paper_mode": paper_mode,
        },
    )

    df_indicators = calculate_base_indicators(hist_df)
    if len(df_indicators) > 1:
        latest_indicators = df_indicators.iloc[-2].to_dict()
    else:
        latest_indicators = df_indicators.iloc[-1].to_dict()
    bootstrap_price = float(hist_df.iloc[-1]['close'])
    chart_tf = str(ui_cfg.get("chart_tf", cfg.get("timeframe", "5m")) or cfg.get("timeframe", "5m"))
    chart_hist_df = _runtime_fetch_ohlcv(
        market,
        symbol,
        chart_tf,
        200,
        paper_mode=paper_mode,
        logger=logger,
    )
    if chart_hist_df is None or chart_hist_df.empty:
        chart_hist_df = hist_df
    chart_hist_df = _closed_chart_candles(chart_hist_df, chart_tf, include_live=True)
    if not chart_hist_df.empty:
        chart_hist_df = calculate_base_indicators(chart_hist_df)
    # Initialize chart_bars with history for dashboard
    candle_limit = (
        int(cfg.get("dashboard", {}).get("candle_limit", 240)) or 240
    )
    chart_bars = deque(maxlen=candle_limit)
    for _, row in chart_hist_df.iterrows():
        chart_bars.append(_chart_bar_payload(row))

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
    auto_learning_min_trades = max(3, int(auto_learning_cfg.get("min_completed_trades", 5)))
    auto_learning_refresh_trades = max(1, int(auto_learning_cfg.get("refresh_closed_trades", 3)))
    auto_learning_shrinkage = float(auto_learning_cfg.get("shrinkage", 0.45))
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
    executor.scale_in_enabled = bool(exec_cfg.get('scale_in_enabled', False))
    executor.scale_in_max_steps = int(exec_cfg.get('scale_in_max_steps', 2))
    executor.scale_in_min_pnl_pct = float(exec_cfg.get('scale_in_min_pnl_pct', 0.0020))
    executor.scale_in_cooldown_seconds = int(exec_cfg.get('scale_in_cooldown_seconds', 180))
    executor.scale_in_position_pct = float(exec_cfg.get('scale_in_position_pct', 0.5))
    executor.scale_in_max_exposure_pct = float(exec_cfg.get('scale_in_max_exposure_pct', 0.50))
    executor.scale_in_wall_buffer_pct = float(exec_cfg.get('scale_in_wall_buffer_pct', 0.002))
    executor.psar_exit_min_streak = int(exec_cfg.get('psar_exit_min_streak', 2))

    if executor.dynamic_leverage_enabled:
        logger.info(f"Dynamic Leverage ENABLED: {executor.leverage_min:.1f}x-{executor.leverage_max:.1f}x (confidence-based)")

    if auto_learning_enabled:
        logger.info("Auto-learning: session-only mode — weights start from defaults, no historical CSV data.")
        _ml_weights = os.path.join(os.path.dirname(__file__) or ".", "..", "ml", "weights.json")
        if os.path.exists(_ml_weights):
            try:
                os.remove(_ml_weights)
                logger.info("Auto-learning: cleared stale weights.json for fresh session start.")
            except Exception:
                pass

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
    state_fetch_ts = 0.0
    last_indicator_fetch_ts = 0.0
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
    last_chart_tf = chart_tf

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
                    new_hist = _closed_chart_candles(new_hist, ui_tf, include_live=True)
                    if not new_hist.empty:
                        new_hist = calculate_base_indicators(new_hist)
                    if not new_hist.empty:
                        chart_bars.clear()
                        for _, row in new_hist.iterrows():
                            chart_bars.append(_chart_bar_payload(row))
                        last_chart_tf = ui_tf
                        status(f"Chart TF: {ui_tf} Loaded")
                    else:
                        status(f"Chart TF {ui_tf}: fetch returned empty data, retrying next cycle")
                except Exception as e:
                    logger.error(f"Failed to switch chart TF: {e}")

            if ticks == 1 or ticks % indicator_refresh_interval == 0:
                logger.info("Refreshing macro indicators...")
                status("Refreshing indicators/MTF")

                # Build the list of OHLCV fetches to run in parallel
                need_pivots = time.time() - last_pivot_refresh_ts >= 900 or not pivot_data
                mtf_enabled = mtf_cfg.get("enabled", False)
                mtf_timeframes = mtf_cfg.get("timeframes", ["15m", "3h", "4h"]) if mtf_enabled else []

                fetch_tasks = {"main": (cfg['timeframe'], 200), "chart": (ui_tf, 5)}
                if need_pivots:
                    fetch_tasks["pivot"] = ("1d", 5)
                for tf in mtf_timeframes:
                    fetch_tasks[f"mtf_{tf}"] = (tf, 200)

                def _fetch_one(key, tf, limit):
                    return key, _runtime_fetch_ohlcv(market, symbol, tf, limit, paper_mode=paper_mode, logger=logger)

                fetched = {}
                with ThreadPoolExecutor(max_workers=min(len(fetch_tasks), 8)) as pool:
                    futures = {pool.submit(_fetch_one, k, tf, lim): k for k, (tf, lim) in fetch_tasks.items()}
                    for fut in as_completed(futures):
                        try:
                            key, df = fut.result()
                            fetched[key] = df
                        except Exception as e:
                            logger.warning(f"Parallel OHLCV fetch error: {e}")

                new_df = fetched.get("main", pd.DataFrame())

                # Chart update
                try:
                    chart_update_df = fetched.get("chart", pd.DataFrame())
                    chart_update_df = _closed_chart_candles(chart_update_df, ui_tf, include_live=True)
                    updated = False
                    if not chart_update_df.empty:
                        bar_map = {b["time"]: i for i, b in enumerate(chart_bars)}
                        for _, row in chart_update_df.iterrows():
                            new_bar = _chart_bar_payload(row)
                            if new_bar["time"] in bar_map:
                                chart_bars[bar_map[new_bar["time"]]] = new_bar
                                updated = True
                            else:
                                chart_bars.append(new_bar)
                                bar_map[new_bar["time"]] = len(chart_bars) - 1
                                updated = True
                    if updated and chart_bars:
                        combined = pd.DataFrame(list(chart_bars))
                        if not combined.empty:
                            combined = combined.sort_values("time").drop_duplicates(subset=["time"], keep="last")
                            combined["timestamp"] = pd.to_datetime(combined["time"], unit="ms", utc=True)
                            combined = calculate_base_indicators(combined)
                            chart_bars.clear()
                            for _, row in combined.iterrows():
                                chart_bars.append(_chart_bar_payload(row))
                except Exception as e:
                    logger.warning(f"Chart periodic update failed: {e}")

                # MTF context
                if mtf_enabled:
                    for tf in mtf_timeframes:
                        htf_df = fetched.get(f"mtf_{tf}", pd.DataFrame())
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
                    last_indicator_fetch_ts = time.time()

                # Pivot points (fetched in parallel above if needed)
                if need_pivots:
                    try:
                        daily_df = fetched.get("pivot", pd.DataFrame())
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
            state_fetch_ts = time.time()

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
            if bool(cfg.get("execution", {}).get("panic_exit", False)):
                logger.warning("Emergency exit requested from dashboard/config.")
                try:
                    if hasattr(executor, "emergency_close_all"):
                        executor.emergency_close_all(symbol)
                    else:
                        executor.close_all_positions(symbol)
                except Exception as e:
                    logger.error(f"Emergency exit failed: {e}")
                finally:
                    cfg.setdefault("execution", {})["panic_exit"] = False
                continue

            if is_paused:
                if bool(cfg.get("execution", {}).get("close_positions_on_pause", False)):
                    if getattr(executor, "active_positions", None) or getattr(executor, "pending_entry", None) or getattr(executor, "pending_exit", None):
                        logger.warning("Bot paused with close_positions_on_pause=true - closing positions and orders.")
                        try:
                            executor.close_all_positions(symbol)
                        except Exception as e:
                            logger.error(f"Error closing positions on pause: {e}")
                else:
                    if getattr(executor, "pending_entry", None):
                        try:
                            order_id = str((getattr(executor, "pending_entry", {}) or {}).get("order_id") or "")
                            if order_id:
                                executor.exchange.cancel_order(order_id, symbol)
                            executor.pending_entry = None
                            logger.info("Bot paused - cancelled pending entry; active positions remain managed.")
                        except Exception as e:
                            logger.warning(f"Paused pending-entry cancel skipped: {e}")
                    try:
                        if hasattr(executor, "_cancel_non_reduce_open_orders"):
                            executor._cancel_non_reduce_open_orders(symbol)
                    except Exception as e:
                        logger.debug(f"Paused non-reduce order cleanup skipped: {e}")

            # Update executor with current ATR for volatility-based leverage
            atr_pct = latest_indicators.get('atr_pct')
            if atr_pct is not None and pd.notna(atr_pct):
                executor._current_atr_pct = float(atr_pct)

            # Pass PSAR to executor for dynamic trailing stop logic
            psar = latest_indicators.get('psar')
            if psar is not None and pd.notna(psar):
                executor._current_psar = float(psar)
            psar_streak = latest_indicators.get('psar_streak')
            if psar_streak is not None and pd.notna(psar_streak):
                executor._current_psar_streak = int(psar_streak)

            if auto_learning_enabled:
                closed_trades = int(getattr(executor, "stats_trades", 0) or 0)
                if closed_trades >= last_learning_closed_trades + auto_learning_refresh_trades:
                    try:
                        from ml import optimize_weights

                        session_trades = getattr(executor, "get_session_trades", lambda: [])()

                        learning_state = optimize_weights(
                            min_trades=auto_learning_min_trades,
                            shrinkage=auto_learning_shrinkage,
                            ai_enabled=auto_learning_ai_enabled,
                            ai_model=auto_learning_ai_model,
                            ai_max_weight_shift=auto_learning_ai_max_shift,
                            quiet=True,
                            session_trades=session_trades,
                        )
                        last_learning_closed_trades = closed_trades
                        if learning_state:
                            executor.learning_risk_multiplier = float(learning_state.get("risk_multiplier", 1.0) or 1.0)
                            wr = float(learning_state.get("win_rate", 0.0)) * 100.0
                            risk_mult = float(getattr(executor, "learning_risk_multiplier", 1.0))
                            status(f"Session learning: WR {wr:.1f}%, risk {risk_mult:.2f}x, {learning_state.get('completed_trades',0)} trades")
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
            base_min_conf = float(runtime_strategy_config.get("min_conf", 0.15) or 0.15)
            runtime_strategy_config["min_conf"] = max(base_min_conf, 0.15)
            runtime_strategy_config["entry_min_confidence_hard"] = max(
                float(runtime_strategy_config.get("entry_min_confidence_hard", 0.20) or 0.20),
                0.20,
            )
            closed_snapshot = list(getattr(executor, "closed_trades", []) or [])[-10:]
            consec_losses = _count_consecutive_losses(closed_snapshot)
            tilt_result = compute_loss_tilt_override(consec_losses, base_min_conf, strategy_config)
            runtime_strategy_config.update(tilt_result["overrides"])
            if tilt_result["pauses"]["should_pause"]:
                loss_tilt_pause_until = max(loss_tilt_pause_until, tilt_result["pauses"]["tilt_pause_until"])
                loss_tilt_last_count = consec_losses
            else:
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

            signal = apply_loss_tilt_pause(signal, loss_tilt_pause_until)

            signal = apply_scalp_hold_guard(signal, executor, exec_cfg)

            if paper_mode:
                signal = apply_aggressive_scalp_gate(signal, state, exec_cfg, runtime_strategy_config, executor)

            raw_score = float(signal.get('score', 0.0) or 0.0)
            signal_history.append(raw_score)

            signal['score'] = raw_score
            signal['confidence'] = abs(max(-1.0, min(1.0, raw_score)))

            min_conf_floor = float(runtime_strategy_config.get("min_conf", 0.10))
            signal = apply_confidence_floor(signal, min_conf_floor, runtime_strategy_config)

            hard_min = float(runtime_strategy_config.get("entry_min_confidence_hard", 0.0) or 0.0)
            _is_aggressive = bool(signal.get("aggressive_scalp", False))
            if not _is_aggressive:
                signal = apply_loss_tilt_hard_gate(signal, hard_min, raw_score, runtime_strategy_config)

            min_rr = float(runtime_strategy_config.get("min_reward_risk", 0.0) or 0.0)
            if min_rr > 0.0:
                signal = apply_reward_risk_gate(signal, min_rr)

            sig_str = f"Signal: {signal.get('action','?')} conf={float(signal.get('confidence',0.0) or 0.0):.1%} Reason: {signal.get('reason','N/A')}"
            if sig_str != last_reported_signal:
                status(sig_str)
                logger.info(f"ANALYSIS: {sig_str}")
                last_reported_signal = sig_str

            if signal.get("action") == "HOLD":
                gate_text = format_gate_trace(signal, ai_overlay_state, is_paused, runtime_strategy_config, loss_tilt_pause_until)
                if gate_text:
                    signal["gate_trace"] = gate_text
                    gate_msg = f"Gate: {gate_text}"
                    if gate_msg != last_reported_status:
                        status(gate_msg)
                        logger.info(f"ANALYSIS: {gate_msg}")
                        last_reported_status = gate_msg

            if bool(ai_overlay_cfg.get("enabled", False)):
                signal = apply_ai_overlay(signal, ai_overlay_state, executor, symbol, status)

            signal, last_ai_trade_ts, last_ai_trade_key, last_ai_trade_resp = apply_ai_trade_gate(
                signal, ai_orch, ai_trade_cfg, ai_model, executor, state, mtf_context,
                runtime_strategy_config, ai_overlay_state, last_ai_trade_ts, last_ai_trade_key,
                last_ai_trade_resp, symbol, status,
            )

            if signal['action'] != "HOLD":
                signal = apply_regime_veto(signal, is_ai_enabled, regime)

            executor._current_signal_snapshot = dict(signal or {})
            executor.process_orders_and_positions(symbol, state['price'])

            # Re-check action after vetoes
            if not is_paused and signal['action'] != "HOLD":
                dispatch_entry(signal, ai_overlay_state, state, executor, symbol)

            curr_status = str(getattr(executor, 'last_status', '') or "")
            if curr_status != last_reported_status:
                status(f"Exec: {curr_status}")
                last_reported_status = curr_status

            # Render Dashboard
            if ticks % 5 == 0:
                open_orders_cache = executor.get_open_orders(symbol)
            ui_mode = _detect_ui_mode(str(ui_cfg.get("mode", "auto")))
            ui_cfg["paused"] = is_paused # Pass pause state to UI
            now = time.time()
            freshness = {
                "live_price_age": now - state_fetch_ts,
                "signal_candle_age": now - last_indicator_fetch_ts if last_indicator_fetch_ts > 0 else -1,
                "mtf_ages": {
                    tf: now - entry.get("computed_at", 0)
                    for tf, entry in mtf_context.items()
                    if isinstance(entry, dict) and entry.get("computed_at", 0) > 0
                },
            }
            print_dashboard(
                ticks, symbol, regime, state, signal, executor, session_start,
                mtf_context=mtf_context, mtf_cfg=mtf_cfg,
                status_lines=list(status_buf)[-int(ui_cfg.get("status_lines", 3) or 3):],
                ui_cfg=ui_cfg, ai_overlay=ai_overlay_state, pivot_data=pivot_data,
                open_orders=open_orders_cache, freshness=freshness
            )

            if dashboard_runtime is not None:
                if ticks == 1 or ticks % 20 == 0:
                    dashboard_runtime.ensure_running(dashboard_host, dashboard_port)
                dashboard_runtime.update_state(
                    _build_dashboard_snapshot(
                        symbol, regime, state, signal, executor, session_start,
                        list(status_buf)[-int(ui_cfg.get("status_lines", 3) or 3):],
                        pivot_data, mtf_context, open_orders_cache, latest_indicators,
                        chart_bars, ai_overlay_state, cfg, freshness=freshness
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

