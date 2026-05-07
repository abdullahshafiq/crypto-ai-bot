"""Smoke test: validates bot imports, config loading, synthetic data pipeline,
indicator calculation, and signal generation — all without network calls."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["BOT_CONFIG"] = "config.paper.test.yaml"


def main():
    print("=" * 60)
    print("BOT STARTUP SMOKETEST")
    print("=" * 60)

    # ── Phase 1: Import chain ──────────────────────────────────
    print("\n[1/6] Testing import chain...")

    from dotenv import load_dotenv

    load_dotenv()

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
    from ui.terminal import print_dashboard, YELLOW, BOLD, RESET
    from ui.windows_vt import (
        _enable_windows_vt_mode,
        _show_cursor_ansi,
        _exit_alt_screen_ansi,
        _detect_ui_mode,
    )

    print("  All imports OK")

    # ── Phase 2: Config loading ─────────────────────────────────
    print("\n[2/6] Testing config loading...")

    cfg = load_config("config.paper.test.yaml")
    symbol = cfg["symbol"]
    mode = cfg["execution"]["mode"]
    tf = cfg["timeframe"]
    print(f"  Config: symbol={symbol}, mode={mode}, tf={tf}")

    # ── Phase 3: MarketData init ───────────────────────────────
    print("\n[3/6] Testing MarketData initialization...")

    data_market = resolve_data_market(cfg)
    market = MarketData(market=data_market)
    print(f"  data_market={data_market}, MarketData OK")

    # ── Phase 4: Synthetic data pipeline ───────────────────────
    print("\n[4/6] Testing synthetic data pipeline...")

    from core.synthetic_data import _fallback_bootstrap_ohlcv

    df = _fallback_bootstrap_ohlcv(symbol, tf, limit=200)
    print(f"  Synthetic: {len(df)} rows, {len(df.columns)} cols")
    assert len(df) >= 50, f"Too few candles: {len(df)}"
    assert "close" in df.columns, "Missing 'close' column"
    assert "volume" in df.columns, "Missing 'volume' column"

    # ── Phase 5: Indicator calculation ─────────────────────────
    print("\n[5/6] Testing indicator calculation...")

    df_ind = calculate_base_indicators(df)
    print(f"  Indicators: {len(df_ind)} rows, {len(df_ind.columns)} cols")
    assert len(df_ind) > 10, f"Indicator rows too few: {len(df_ind)}"

    latest = df_ind.iloc[-1].to_dict()

    # ── Phase 6: Signal generation ─────────────────────────────
    print("\n[6/6] Testing signal generation...")

    import pandas as pd

    strategy_config = dict(cfg.get("strategy", {}) or {})
    strategy_config.setdefault("timeframe", tf)
    strategy_config.setdefault("max_spread", 0.0005)
    strategy_config.setdefault("min_conf", 0.15)

    state = {
        "price": float(latest.get("close", 100.0)),
        "spread_pct": 0.0001,
        "ask": float(latest.get("close", 100.0)) * 1.00005,
        "bid": float(latest.get("close", 100.0)) * 0.99995,
        "volume_state": "NORMAL",
        "ret_30s": 0.0,
        "ret_5s": 0.0,
    }

    signal = generate_quant_signal(
        state,
        latest,
        strategy_config,
        df_ind,
        None,
        mtf_context={},
        mtf_config=cfg.get("mtf", {}) or {},
        pivot_data={},
    )

    action = signal.get("action", "UNKNOWN")
    score = signal.get("score", 0.0)
    confidence = signal.get("confidence", 0.0)

    print(f"  Signal: action={action}, score={score:.4f}, conf={confidence:.4f}")
    assert action in ("BUY", "SELL", "HOLD"), f"Unexpected action: {action}"

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL SMOKETESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()