import csv
import os


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def compute_trade_metrics(log_file: str = "trade_log_futures.csv") -> dict:
    if not os.path.exists(log_file):
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0}
    rows = []
    with open(log_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = str(row.get("event", "")).upper()
            if event.startswith("EXIT") or event in {"TAKE_PROFIT", "TRAIL_WIN", "TRAIL_SL", "MANUAL_CLOSE"}:
                pnl = _to_float(row.get("pnl", 0.0))
                fees = _to_float(row.get("fees", 0.0))
                rows.append(pnl - fees)
    if not rows:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0}
    gross_profit = sum(x for x in rows if x > 0)
    gross_loss = abs(sum(x for x in rows if x < 0))
    wins = sum(1 for x in rows if x > 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in rows:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return {
        "trades": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0),
        "expectancy": sum(rows) / len(rows),
        "max_drawdown": max_dd,
    }


def paper_gate_passed(log_file: str = "trade_log_futures.csv", min_trades: int = 100, min_profit_factor: float = 1.2, max_drawdown: float = 0.20) -> tuple[bool, dict]:
    metrics = compute_trade_metrics(log_file=log_file)
    passed = (
        int(metrics["trades"]) >= int(min_trades)
        and float(metrics["profit_factor"]) >= float(min_profit_factor)
        and float(metrics["expectancy"]) > 0.0
        and float(metrics["max_drawdown"]) <= float(max_drawdown)
    )
    return passed, metrics
