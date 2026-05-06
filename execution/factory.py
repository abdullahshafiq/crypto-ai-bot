import os

from .futures import BinanceFuturesExecution
from .spot import BinanceSpotExecution

SPOT_TRADE_LOG_FILE = "trade_log_spot.csv"
FUTURES_TRADE_LOG_FILE = "trade_log_futures.csv"


def resolve_data_market(cfg: dict) -> str:
    data_cfg = cfg.get("data", {}) or {}
    configured = str(data_cfg.get("market", "auto") or "auto").strip().lower()
    if configured in {"usdm", "spot"}:
        return configured
    exec_market = str((cfg.get("execution", {}) or {}).get("market", "usdm") or "usdm").strip().lower()
    return "spot" if exec_market == "spot" else "usdm"


def apply_common_executor_settings(executor, cfg: dict, fixed_trade_usdt: float, fee_rate: float):
    exec_cfg = cfg.get("execution", {}) or {}
    strategy_cfg = cfg.get("strategy", {}) or {}
    spot_cfg = cfg.get("spot", {}) or {}
    executor.fee_rate = float(fee_rate)
    executor.fee_slippage_buffer_pct = float(exec_cfg.get("fee_slippage_buffer_pct", 0.0002))
    executor.fee_edge_multiplier = float(exec_cfg.get("fee_edge_multiplier", 1.2))
    executor.fixed_trade_usdt = float(fixed_trade_usdt)
    executor.tp_pct = float(strategy_cfg.get("tp_pct", getattr(executor, "tp_pct", 0.0025)))
    if hasattr(executor, "same_side_reentry_cooldown_seconds"):
        executor.same_side_reentry_cooldown_seconds = int(
            exec_cfg.get(
                "same_side_reentry_cooldown_seconds",
                getattr(executor, "same_side_reentry_cooldown_seconds", 180),
            )
        )
    if hasattr(executor, "same_side_reentry_strong_confidence"):
        executor.same_side_reentry_strong_confidence = float(
            exec_cfg.get(
                "same_side_reentry_strong_confidence",
                getattr(executor, "same_side_reentry_strong_confidence", 0.85),
            )
        )
    if hasattr(executor, "scalp_config"):
        scalp_cfg = dict(getattr(executor, "scalp_config", {}) or {})
        scalp_cfg["runner_enabled"] = bool(exec_cfg.get("scalp_runner_enabled", scalp_cfg.get("runner_enabled", True)))
        scalp_cfg["tp_pct"] = float(strategy_cfg.get("tp_pct", scalp_cfg.get("tp_pct", 0.0025)))
        scalp_cfg["min_hold_seconds"] = int(exec_cfg.get("scalp_min_hold_seconds", scalp_cfg.get("min_hold_seconds", 10)))
        scalp_cfg["runner_pullback_pct"] = float(exec_cfg.get("scalp_runner_pullback_pct", scalp_cfg.get("runner_pullback_pct", 0.0012)))
        scalp_cfg["runner_min_lock_pct"] = float(exec_cfg.get("scalp_runner_min_lock_pct", scalp_cfg.get("runner_min_lock_pct", 0.0018)))
        scalp_cfg["runner_exchange_tp_multiplier"] = float(exec_cfg.get("scalp_runner_exchange_tp_multiplier", scalp_cfg.get("runner_exchange_tp_multiplier", 3.0)))
        scalp_cfg["runner_partial_exit_pct"] = float(exec_cfg.get("scalp_runner_partial_exit_pct", scalp_cfg.get("runner_partial_exit_pct", 0.45)))
        executor.scalp_config = scalp_cfg
    executor.default_sl_pct = float(strategy_cfg.get("sl_pct", getattr(executor, "default_sl_pct", 0.0030)))
    executor.min_seconds_between_trades = max(60, int(exec_cfg.get("min_seconds_between_trades", 60)))
    executor.min_seconds_before_reversal = max(60, int(exec_cfg.get("min_seconds_before_reversal", 60)))
    executor.reversal_min_confidence = max(0.45, float(exec_cfg.get("reversal_min_confidence", 0.45)))
    executor.reversal_min_score = max(0.25, float(exec_cfg.get("reversal_min_score", 0.25)))
    executor.reversal_min_net_edge_pct = float(exec_cfg.get("reversal_min_net_edge_pct", 0.0015))
    executor.break_even_trigger_pct = float(exec_cfg.get("break_even_trigger_pct", 0.0030))
    executor.break_even_buffer_pct = float(exec_cfg.get("break_even_buffer_pct", 0.0004))
    executor.profit_trailing_enabled = bool(exec_cfg.get("profit_trailing_enabled", getattr(executor, "profit_trailing_enabled", True)))
    executor.profit_trailing_activation_pct = float(
        exec_cfg.get("profit_trailing_activation_pct", getattr(executor, "profit_trailing_activation_pct", executor.break_even_trigger_pct))
    )
    executor.trailing_tp_enabled = bool(exec_cfg.get("trailing_tp_enabled", getattr(executor, "trailing_tp_enabled", True)))
    executor.trailing_tp_giveback_pct = float(exec_cfg.get("trailing_tp_giveback_pct", getattr(executor, "trailing_tp_giveback_pct", 0.12)))
    executor.trailing_tp_min_peak_pct = float(
        exec_cfg.get("trailing_tp_min_peak_pct", getattr(executor, "trailing_tp_min_peak_pct", executor.profit_trailing_activation_pct))
    )
    executor.ttl_exit_only_if_unprofitable = bool(
        exec_cfg.get("ttl_exit_only_if_unprofitable", getattr(executor, "ttl_exit_only_if_unprofitable", True))
    )
    executor.ttl_exit_profit_cap_pct = float(
        exec_cfg.get("ttl_exit_profit_cap_pct", getattr(executor, "ttl_exit_profit_cap_pct", 0.0))
    )
    executor.trail_tighten_1_pct = float(exec_cfg.get("trail_tighten_1_pct", 0.0050))
    executor.trail_tighten_2_pct = float(exec_cfg.get("trail_tighten_2_pct", 0.0100))
    executor.trail_t1_gap_pct = float(exec_cfg.get("trail_t1_gap_pct", getattr(executor, "trail_t1_gap_pct", 0.0025)))
    executor.trail_t2_gap_pct = float(exec_cfg.get("trail_t2_gap_pct", getattr(executor, "trail_t2_gap_pct", 0.0020)))
    executor.min_profit_after_fees = float(exec_cfg.get("min_profit_after_fees", 0.0012))
    executor.exit_on_reversal_only_in_profit = bool(exec_cfg.get("exit_on_reversal_only_in_profit", True))
    executor.use_limit_orders = bool(exec_cfg.get("use_limit_orders", True))
    executor.use_native_trailing_stop = bool(exec_cfg.get("use_native_trailing_stop", False))
    executor.use_exchange_stop_loss = bool(exec_cfg.get("use_exchange_stop_loss", True))
    executor.use_exchange_take_profit = bool(exec_cfg.get("use_exchange_take_profit", True))
    executor.market_fallback_on_timeout = bool(exec_cfg.get("market_fallback_on_timeout", False))
    executor.pending_entry_ttl_seconds = int(exec_cfg.get("pending_entry_ttl_seconds", getattr(executor, "pending_entry_ttl_seconds", 20)))
    executor.resting_entry_ttl_seconds = int(exec_cfg.get("resting_entry_ttl_seconds", getattr(executor, "resting_entry_ttl_seconds", 120)))
    executor.ttl_exit_seconds = int(exec_cfg.get("ttl_exit_seconds", getattr(executor, "ttl_exit_seconds", 0)))
    trailing_callback_pct = float(exec_cfg.get("trailing_callback_pct", 0.6))
    executor.trailing_callback_pct = trailing_callback_pct
    executor.trailing_stop_callback = trailing_callback_pct / 100.0
    default_trade_log = SPOT_TRADE_LOG_FILE if str(exec_cfg.get("market", "usdm")).lower() == "spot" else FUTURES_TRADE_LOG_FILE
    executor.trade_log_file = str(exec_cfg.get("trade_log_file", default_trade_log) or default_trade_log)
    executor.spot_balance_pct = float(exec_cfg.get("spot_balance_pct", getattr(executor, "spot_balance_pct", 0.20)))
    executor.spot_reserve_pct = float(exec_cfg.get("spot_reserve_pct", spot_cfg.get("reserve_quote_pct", getattr(executor, "spot_reserve_pct", 0.30))))
    executor.spot_max_layers = int(exec_cfg.get("spot_max_layers", spot_cfg.get("max_layers", getattr(executor, "spot_max_layers", 3))))
    executor.spot_mode = str(spot_cfg.get("mode", getattr(executor, "spot_mode", "grid")) or "grid")
    executor.layer_quote_pct = float(spot_cfg.get("layer_quote_pct", getattr(executor, "layer_quote_pct", 0.20)))
    executor.reserve_quote_pct = float(spot_cfg.get("reserve_quote_pct", getattr(executor, "reserve_quote_pct", 0.30)))
    executor.buy_near_support_pct = float(spot_cfg.get("buy_near_support_pct", getattr(executor, "buy_near_support_pct", 0.0020)))
    executor.sell_near_resistance_pct = float(spot_cfg.get("sell_near_resistance_pct", getattr(executor, "sell_near_resistance_pct", 0.0020)))
    executor.layer_spacing_pct = float(spot_cfg.get("layer_spacing_pct", getattr(executor, "layer_spacing_pct", 0.0030)))
    executor.emergency_break_pct = float(spot_cfg.get("emergency_break_pct", getattr(executor, "emergency_break_pct", 0.0040)))
    executor.min_take_profit_pct = float(spot_cfg.get("min_take_profit_pct", getattr(executor, "min_take_profit_pct", 0.0035)))
    executor.max_spot_layers = executor.spot_max_layers


def create_executor(cfg: dict, api_key: str | None, api_secret: str | None, bootstrap_price: float, fixed_trade_usdt: float):
    exec_cfg = cfg.get("execution", {}) or {}
    exec_mode = str(exec_cfg.get("mode", os.getenv("EXECUTION_MODE", "live"))).strip().lower()
    exec_market = str(exec_cfg.get("market", "usdm") or "usdm").strip().lower()
    leverage = int(exec_cfg.get("leverage", 5))
    mem_cfg = cfg.get("memory", {}) or {}
    fee_rate_cfg = float(exec_cfg.get("fee_rate", 0.0006))
    fee_rate = max(fee_rate_cfg, 0.0)

    is_paper = (exec_mode == "paper")
    if not is_paper and exec_mode != "live":
        raise RuntimeError(f"Unknown execution.mode={exec_mode!r}. Use 'live' or 'paper'.")

    if exec_market == "spot":
        executor = BinanceSpotExecution(
            api_key,
            api_secret,
            max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
            is_demo=is_paper,
        )
    elif is_paper:
        from .paper import PaperFuturesExecution
        executor = PaperFuturesExecution(
            starting_balance_usdt=float(exec_cfg.get("paper_starting_balance_usdt", 1000)),
            leverage=leverage,
            fee_rate=fee_rate,
            max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
        )
        executor.symbol = cfg.get("symbol", "SOL/USDC:USDC")
    else:
        if not api_key or not api_secret:
            raise RuntimeError("Live mode requires real BINANCE_API_KEY and BINANCE_SECRET.")
        executor = BinanceFuturesExecution(
            api_key,
            api_secret,
            symbol=cfg.get("symbol", "SOL/USDC:USDC"),
            leverage=leverage,
            max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
            is_demo=False,
        )
        _ = executor.get_portfolio_value(bootstrap_price)
        if float(getattr(executor, "initial_balance", 0.0) or 0.0) <= 0:
            raise RuntimeError("Futures executor could not confirm account equity; refusing to trade.")

    apply_common_executor_settings(executor, cfg, fixed_trade_usdt=fixed_trade_usdt, fee_rate=fee_rate)
    return executor
