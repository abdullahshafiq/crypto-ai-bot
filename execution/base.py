import csv
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

TRADE_LOG_FILE = "trade_log_futures.csv"
TRADE_LOG_HEADER = [
    "timestamp",
    "trade_id",
    "event",
    "side",
    "price",
    "amount",
    "pnl",
    "fees",
    "score",
    "confidence",
    "reason",
    "type",
]


def _next_trade_id_from_log(log_file: str, default: int = 1) -> int:
    """Return one greater than the highest trade_id already present in a log."""
    try:
        if not os.path.exists(log_file):
            return int(default)
        max_id = 0
        with open(log_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    max_id = max(max_id, int(float(row.get("trade_id", 0) or 0)))
                except (TypeError, ValueError):
                    continue
        return max(int(default), max_id + 1)
    except Exception as e:
        logger.debug(f"Trade id bootstrap skipped for {log_file}: {e}")
        return int(default)


def _normalize_futures_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip()
    if not symbol:
        return symbol
    if ":" in symbol:
        return symbol
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        quote = quote.split(":")[0]
        return f"{base}/{quote}:{quote}"
    return symbol


def _normalize_spot_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip()
    if not symbol:
        return symbol
    if ":" in symbol:
        base, quote = symbol.split(":", 1)[0].split("/", 1)
        return f"{base}/{quote}"
    return symbol


def _market_id_from_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace(":USDT", "").replace(":USDC", "")


def _quote_asset_from_symbol(symbol: str) -> str:
    symbol = _normalize_futures_symbol(symbol or "DOGE/USDT")
    if "/" not in symbol:
        return "USDT"
    quote = symbol.split("/", 1)[1].split(":", 1)[0].upper()
    return quote or "USDT"


def _order_fill_price(order: dict, fallback: float) -> float:
    if not isinstance(order, dict):
        return float(fallback)
    for key in ("average", "price"):
        value = order.get(key)
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def _maker_entry_price(exchange, symbol: str, side: str, fallback: float) -> float:
    """
    Place maker entries close to the book without crossing it.
    BUY: near best bid
    SELL: near best ask
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=5) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        best_bid = float(bids[0][0]) if bids else float(fallback)
        best_ask = float(asks[0][0]) if asks else float(fallback)
        tick = max(best_bid * 0.0001, fallback * 0.00005, 0.0000001)
        side_u = str(side or "").upper()

        if side_u == "BUY":
            price = min(best_bid + tick, best_ask - tick) if best_ask > best_bid else best_bid + tick
            return float(price)

        price = max(best_ask - tick, best_bid + tick) if best_ask > best_bid else best_ask - tick
        return float(price)
    except Exception:
        return float(fallback)


def _realized_exit_type(exit_type: str, net_pnl: float) -> str:
    base = str(exit_type or "EXIT").upper()
    if base in {"TRAIL_WIN", "TRAIL_SL"}:
        return "TRAIL_WIN" if float(net_pnl) > 0 else "TRAIL_SL"
    if base in {"TAKE_PROFIT", "TRAIL_TP", "SCALP_EXIT", "SCALP_EXIT_PARTIAL", "REVERSAL_BANK", "REVERSAL", "REVERSAL_WIN", "REVERSAL_CUT"}:
        return base if float(net_pnl) > 0 else f"{base}_LOSS"
    return base if float(net_pnl) > 0 else f"{base}_LOSS"


def _exchange_flag_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _order_id(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    info = order.get("info", {}) or {}
    return str(order.get("id") or order.get("orderId") or info.get("orderId") or "")


def _order_type(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    info = order.get("info", {}) or {}
    value = str(
        order.get("origType")
        or info.get("origType")
        or order.get("type")
        or info.get("type")
        or ""
    ).upper().replace(" ", "_")
    if value in {"STOP", "STOP_MARKET"}:
        return "STOP_MARKET"
    if value in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
        return "TAKE_PROFIT_MARKET"
    if value in {"TRAILING_STOP", "TRAILING_STOP_MARKET"}:
        return "TRAILING_STOP_MARKET"
    return value


def _order_trigger_price(order: dict) -> float:
    if not isinstance(order, dict):
        return 0.0
    info = order.get("info", {}) or {}
    for key in ("stopPrice", "activatePrice", "activationPrice", "triggerPrice", "price"):
        for source in (order, info):
            try:
                value = float(source.get(key))
                if value > 0:
                    return value
            except (TypeError, ValueError, AttributeError):
                continue
    return 0.0


def _default_sl_price(side: str, entry: float, sl_pct: float) -> float:
    side = str(side or "").upper()
    entry = float(entry)
    sl_pct = abs(float(sl_pct or 0.0030))
    return entry * (1 - sl_pct) if side in {"LONG", "BUY"} else entry * (1 + sl_pct)


def _pivot_guarded_sl_price(side: str, entry: float, sl: float, pivot_classic=None) -> float:
    side_u = str(side or "").upper()
    if side_u not in {"SHORT", "SELL"} or not isinstance(pivot_classic, dict):
        return sl
    try:
        r1 = float(pivot_classic.get("r1", 0.0) or 0.0)
    except (TypeError, ValueError):
        r1 = 0.0
    if entry > 0 and r1 > entry and sl < r1 * 1.001:
        return r1 * 1.002
    return sl


def _safe_initial_sl_price(side: str, entry: float, proposed_sl, sl_pct: float, pivot_classic=None) -> float:
    side_u = str(side or "").upper()
    entry = float(entry)
    default_sl = _default_sl_price(side_u, entry, sl_pct)
    default_sl = _pivot_guarded_sl_price(side_u, entry, default_sl, pivot_classic)
    min_dist_pct = abs(float(sl_pct or 0.0030))
    try:
        sl = float(proposed_sl)
    except (TypeError, ValueError):
        return default_sl

    if entry <= 0 or sl <= 0:
        return default_sl

    if side_u in {"LONG", "BUY"}:
        if sl >= entry:
            return default_sl
        if ((entry - sl) / entry) < min_dist_pct:
            return default_sl
    else:
        if sl <= entry:
            return default_sl
        if ((sl - entry) / entry) < min_dist_pct:
            return default_sl

    return _pivot_guarded_sl_price(side_u, entry, sl, pivot_classic)


def _safe_tp_price(side: str, entry: float, proposed_tp=None, tp_pct: float = 0.0025) -> float:
    side_u = str(side or "").upper()
    entry = float(entry or 0.0)
    tp_pct = abs(float(tp_pct or 0.0025))
    if entry <= 0:
        return 0.0

    try:
        tp = float(proposed_tp)
    except (TypeError, ValueError):
        tp = 0.0

    if side_u in {"LONG", "BUY"}:
        if tp > entry:
            return tp
        return entry * (1 + tp_pct)

    if tp > 0 and tp < entry:
        return tp
    return entry * (1 - tp_pct)


def _runner_emergency_tp_price(side: str, entry: float, base_tp: float, scalp_cfg: dict) -> float:
    """Return a farther exchange TP so local runner logic can squeeze profits."""
    if not bool((scalp_cfg or {}).get("runner_enabled", True)):
        return float(base_tp)

    tp_pct = abs(float((scalp_cfg or {}).get("tp_pct", 0.0025) or 0.0025))
    multiplier = max(1.0, float((scalp_cfg or {}).get("runner_exchange_tp_multiplier", 3.0) or 3.0))
    emergency_dist = tp_pct * multiplier
    side_u = str(side or "").upper()
    entry = float(entry)
    base_tp = float(base_tp)

    if side_u in {"LONG", "BUY"}:
        emergency_tp = entry * (1 + emergency_dist)
        return max(base_tp, emergency_tp)
    emergency_tp = entry * (1 - emergency_dist)
    return min(base_tp, emergency_tp)


def _trailing_tp_hit(pos: dict, profit_pct: float, min_net_profit: float) -> bool:
    """Close when current profit gives back enough from the best profit seen."""
    if not bool(pos.get("trailing_tp_enabled", True)):
        return False
    peak_profit = float(pos.get("highest_profit_pct", 0.0) or 0.0)
    min_peak = max(
        float(pos.get("trailing_tp_min_peak_pct", 0.0020) or 0.0020),
        float(pos.get("profit_trailing_activation_pct", 0.0) or 0.0),
        float(min_net_profit),
    )
    if peak_profit < min_peak:
        return False

    giveback_pct = max(0.01, min(0.95, float(pos.get("trailing_tp_giveback_pct", 0.12) or 0.12)))
    floor_profit = peak_profit * (1.0 - giveback_pct)
    pos["trailing_tp_floor_pct"] = floor_profit
    return float(profit_pct) >= float(min_net_profit) and float(profit_pct) <= floor_profit


def _compute_trailing_stop(pos: dict, current_price: float, current_psar: float = None) -> float:
    """
    Progressive trailing stop with 4 stages. Stores 'trail_stage' on pos dict.

    LONG stages:
      BASE  (profit <  be_pct): SL = initial, no trail
      BE+   (profit >= be_pct): SL = entry + fees  (risk-free)
      T1    (profit >= t1_pct): SL = trailing at 0.25% behind best price
      T2    (profit >= t2_pct): SL = trailing at 0.20% behind best price

    SHORT: mirror with inverted signs.
    """
    entry = float(pos.get("entry", current_price) or current_price)
    side = str(pos.get("side", "LONG"))
    base_dist = float(pos.get("sl_pct_dist", 0.005) or 0.005)
    be_pct = float(pos.get("break_even_trigger_pct", 0.0015) or 0.0015)
    be_buffer = float(pos.get("break_even_buffer_pct", 0.0004) or 0.0004)
    profit_only = bool(pos.get("profit_trailing_enabled", True))
    profit_activation = float(pos.get("profit_trailing_activation_pct", be_pct) or be_pct)
    if profit_only:
        profit_activation = max(0.0, profit_activation, be_pct)
    t1_pct = float(pos.get("trail_tighten_1_pct", 0.0025) or 0.0025)
    t2_pct = float(pos.get("trail_tighten_2_pct", 0.0035) or 0.0035)
    fee_rate = float(pos.get("fee_rate", 0.0004) or 0.0004)
    min_profit = float(pos.get("min_profit_after_fees", 0.0002) or 0.0002)

    t1_gap = float(pos.get("trail_t1_gap_pct", 0.0025) or 0.0025)
    t2_gap = float(pos.get("trail_t2_gap_pct", 0.0020) or 0.0020)
    safety_margin = 0.0005

    if side == "LONG":
        best_price = float(pos.get("highest_price", entry) or entry)
        profit_pct = (best_price - entry) / entry if entry else 0.0

        trail_sl = entry * (1 - base_dist)
        stage = "BASE"

        if profit_only and profit_pct < profit_activation:
            pos["trail_stage"] = "WAIT_PROFIT"
            return float(pos.get("sl", trail_sl) or trail_sl)

        if profit_pct >= be_pct:
            be_sl = entry * (1 + (2.0 * fee_rate) + min_profit)
            trail_sl = max(trail_sl, be_sl)
            stage = "BE+"

        if profit_pct >= t1_pct:
            t1_sl = best_price * (1 - t1_gap)
            trail_sl = max(trail_sl, t1_sl)
            stage = "T1"

        if profit_pct >= t2_pct:
            t2_sl = best_price * (1 - t2_gap)
            trail_sl = max(trail_sl, t2_sl)
            stage = "T2"

        trail_sl = min(trail_sl, current_price * (1 - safety_margin))
        pos["trail_stage"] = stage
        return trail_sl

    # SHORT
    best_price = float(pos.get("lowest_price", entry) or entry)
    profit_pct = (entry - best_price) / entry if entry else 0.0

    trail_sl = entry * (1 + base_dist)
    stage = "BASE"

    if profit_only and profit_pct < profit_activation:
        pos["trail_stage"] = "WAIT_PROFIT"
        return float(pos.get("sl", trail_sl) or trail_sl)

    if profit_pct >= be_pct:
        be_sl = entry * (1 - (2.0 * fee_rate) - min_profit)
        trail_sl = min(trail_sl, be_sl)
        stage = "BE+"

    if profit_pct >= t1_pct:
        t1_sl = best_price * (1 + t1_gap)
        trail_sl = min(trail_sl, t1_sl)
        stage = "T1"

    if profit_pct >= t2_pct:
        t2_sl = best_price * (1 + t2_gap)
        trail_sl = min(trail_sl, t2_sl)
        stage = "T2"

    trail_sl = max(trail_sl, current_price * (1 + safety_margin))
    pos["trail_stage"] = stage
    return trail_sl
