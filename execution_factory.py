import os

from futures_execution import BinanceFuturesExecution, PaperFuturesExecution
from spot_execution import BinanceSpotExecution
from execution_shared import FUTURES_TRADE_LOG_FILE, SPOT_TRADE_LOG_FILE


def resolve_data_market(cfg: dict) -> str:
    data_cfg = cfg.get("data", {}) or {}
    configured = str(data_cfg.get("market", "auto") or "auto").strip().lower()
    if configured in {"usdm", "spot"}:
        return configured
    exec_market = str((cfg.get("execution", {}) or {}).get("market", "usdm") or "usdm").strip().lower()
    return "spot" if exec_market == "spot" else "usdm"


def apply_common_executor_settings(executor, cfg: dict, fixed_trade_usdt: float, fee_rate: float):
    exec_cfg = cfg.get("execution", {}) or {}
    executor.fee_rate = float(fee_rate)
    executor.fee_slippage_buffer_pct = float(exec_cfg.get("fee_slippage_buffer_pct", 0.0002))
    executor.fee_edge_multiplier = float(exec_cfg.get("fee_edge_multiplier", 1.2))
    executor.fixed_trade_usdt = float(fixed_trade_usdt)
    executor.min_seconds_between_trades = int(exec_cfg.get("min_seconds_between_trades", 30))
    executor.min_seconds_before_reversal = int(exec_cfg.get("min_seconds_before_reversal", 30))
    executor.reversal_min_confidence = float(exec_cfg.get("reversal_min_confidence", 0.20))
    executor.reversal_min_score = float(exec_cfg.get("reversal_min_score", 0.15))
    executor.reversal_min_net_edge_pct = float(exec_cfg.get("reversal_min_net_edge_pct", 0.0015))
    executor.break_even_trigger_pct = float(exec_cfg.get("break_even_trigger_pct", 0.0030))
    executor.break_even_buffer_pct = float(exec_cfg.get("break_even_buffer_pct", 0.0004))
    executor.trail_tighten_1_pct = float(exec_cfg.get("trail_tighten_1_pct", 0.0050))
    executor.trail_tighten_2_pct = float(exec_cfg.get("trail_tighten_2_pct", 0.0100))
    executor.min_profit_after_fees = float(exec_cfg.get("min_profit_after_fees", 0.0012))
    executor.exit_on_reversal_only_in_profit = bool(exec_cfg.get("exit_on_reversal_only_in_profit", True))
    executor.use_limit_orders = bool(exec_cfg.get("use_limit_orders", True))
    executor.use_native_trailing_stop = bool(exec_cfg.get("use_native_trailing_stop", False))
    trailing_callback_pct = float(exec_cfg.get("trailing_callback_pct", 0.6))
    executor.trailing_callback_pct = trailing_callback_pct
    executor.trailing_stop_callback = trailing_callback_pct / 100.0
    executor.trade_log_file = SPOT_TRADE_LOG_FILE if str(exec_cfg.get("market", "usdm")).lower() == "spot" else FUTURES_TRADE_LOG_FILE


def create_executor(cfg: dict, api_key: str | None, api_secret: str | None, bootstrap_price: float, fixed_trade_usdt: float):
    exec_cfg = cfg.get("execution", {}) or {}
    exec_mode = str(exec_cfg.get("mode", os.getenv("EXECUTION_MODE", "paper"))).strip().lower()
    exec_market = str(exec_cfg.get("market", "usdm") or "usdm").strip().lower()
    leverage = int(exec_cfg.get("leverage", 5))
    mem_cfg = cfg.get("memory", {}) or {}
    fee_rate_cfg = float(exec_cfg.get("fee_rate", 0.0006))
    # Paper mode safety: never allow zero-cost assumptions.
    fee_rate = max(fee_rate_cfg, 0.0006 if exec_mode == "paper" else 0.0)

    if exec_market == "spot":
        if exec_mode in {"demo", "binance", "live"} and api_key and api_secret and "your_testnet" not in str(api_key):
            executor = BinanceSpotExecution(
                api_key,
                api_secret,
                max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
                is_demo=(exec_mode != "live"),
            )
        else:
            # Spot fallback keeps paper futures model for safety/compatibility.
            executor = PaperFuturesExecution(
                starting_balance_usdt=float(exec_cfg.get("paper_starting_balance_usdt", 1000.0)),
                leverage=1,
                fee_rate=fee_rate,
                max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
            )
            executor.label = "PAPER SPOT-SAFE"
    else:
        if exec_mode in {"demo", "binance", "live"} and api_key and api_secret and "your_testnet" not in str(api_key):
            executor = BinanceFuturesExecution(
                api_key,
                api_secret,
                symbol=cfg.get("symbol", "SOL/USDC:USDC"),
                leverage=leverage,
                max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
                is_demo=(exec_mode != "live"),
            )
            _ = executor.get_portfolio_value(bootstrap_price)
            if float(getattr(executor, "initial_balance", 0.0) or 0.0) <= 0:
                executor = PaperFuturesExecution(
                    starting_balance_usdt=float(exec_cfg.get("paper_starting_balance_usdt", 1000.0)),
                    leverage=leverage,
                    fee_rate=fee_rate,
                    max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
                )
        else:
            executor = PaperFuturesExecution(
                starting_balance_usdt=float(exec_cfg.get("paper_starting_balance_usdt", 1000.0)),
                leverage=leverage,
                fee_rate=fee_rate,
                max_closed_trades=int(mem_cfg.get("max_closed_trades", 5000)),
            )

    apply_common_executor_settings(executor, cfg, fixed_trade_usdt=fixed_trade_usdt, fee_rate=fee_rate)
    return executor
