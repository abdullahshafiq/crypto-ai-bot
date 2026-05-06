import copy
import time


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
        trade_pnl = float(trade.get("pnl", 0.0) or 0.0)
        realized_net += trade_pnl
        if trade_pnl >= 0:
            realized_profit += trade_pnl
        else:
            realized_loss += abs(trade_pnl)
    stats_trades = int(getattr(executor, "stats_trades", len(getattr(executor, "closed_trades", []) or [])) or 0)
    stats_wins = int(getattr(executor, "stats_wins", 0) or 0)
    stats_losses = int(getattr(executor, "stats_losses", 0) or 0)
    win_rate = (stats_wins / stats_trades * 100.0) if stats_trades > 0 else 0.0
    # Calculate Unrealized PnL for active positions
    unrealized_pnl = 0.0
    total_active_cost = 0.0
    normalized_positions = []
    pnl_estimate = {
        "active": False,
        "side": "FLAT",
        "entry": 0.0,
        "price": current_price,
        "trade_usdt": 0.0,
        "effective_leverage": float(getattr(executor, "leverage", 1.0) or 1.0),
        "notional_usdt": 0.0,
        "gross_pnl": 0.0,
        "gross_pnl_pct": 0.0,
        "estimated_fees": 0.0,
        "net_pnl": 0.0,
        "net_pnl_pct": 0.0,
        "formula": "LONG: notional × (price / entry - 1); SHORT: notional × (1 - price / entry)",
    }
    for pos in positions:
        entry = float(pos.get("entry", 0.0) or 0.0)
        amount = float(pos.get("amount", 0.0) or 0.0)
        side = str(pos.get("side", "") or "").upper()
        trade_usdt = float(pos.get("trade_usdt", 0.0) or 0.0)
        notional = entry * amount
        if trade_usdt > 0:
            effective_leverage = float(pos.get("effective_leverage", notional / trade_usdt) or (notional / trade_usdt))
        elif notional > 0:
            effective_leverage = float(pos.get("effective_leverage", getattr(executor, "leverage", 1.0)) or getattr(executor, "leverage", 1.0))
            trade_usdt = notional / max(effective_leverage, 1.0)
        else:
            effective_leverage = float(pos.get("effective_leverage", getattr(executor, "leverage", 1.0)) or getattr(executor, "leverage", 1.0))
        pos_pnl = (current_price - entry) if side == "LONG" else (entry - current_price)
        gross_pnl = pos_pnl * amount
        gross_pnl_pct = (gross_pnl / notional * 100.0) if notional > 0 else 0.0
        estimated_fees = notional * float(getattr(executor, "fee_rate", 0.0) or 0.0) * 2.0
        net_pnl = gross_pnl - estimated_fees
        net_pnl_pct = (net_pnl / notional * 100.0) if notional > 0 else 0.0

        pos["entry_price"] = entry
        pos["stop_loss"] = float(pos.get("sl", 0.0) or 0.0)
        pos["take_profit"] = float(pos.get("tp_price", pos.get("tp_target", 0.0)) or 0.0)
        pos["unrealized_pnl"] = gross_pnl
        pos["unrealized_pnl_pct"] = gross_pnl_pct
        pos["trade_usdt"] = trade_usdt
        pos["effective_leverage"] = effective_leverage
        pos["notional_usdt"] = notional
        normalized_positions.append(pos)
        unrealized_pnl += gross_pnl
        total_active_cost += notional
        if not pnl_estimate["active"]:
            pnl_estimate = {
                "active": True,
                "side": side,
                "entry": entry,
                "price": current_price,
                "trade_usdt": trade_usdt,
                "effective_leverage": effective_leverage,
                "notional_usdt": notional,
                "gross_pnl": gross_pnl,
                "gross_pnl_pct": gross_pnl_pct,
                "estimated_fees": estimated_fees,
                "net_pnl": net_pnl,
                "net_pnl_pct": net_pnl_pct,
                "formula": "LONG: notional × (price / entry - 1); SHORT: notional × (1 - price / entry)",
            }
    if not pnl_estimate["active"] and current_price > 0 and isinstance(signal, dict):
        signal_action = str(signal.get("action", "") or "").upper()
        if signal_action in {"BUY", "SELL"}:
            entry = float(signal.get("entry", current_price) or current_price)
            trade_usdt = float(
                cfg.get("strategy", {}).get(
                    "fixed_trade_usdt",
                    getattr(executor, "fixed_trade_usdt", 0.0),
                )
                or getattr(executor, "fixed_trade_usdt", 0.0)
                or 0.0
            )
            effective_leverage = float(getattr(executor, "leverage", 1.0) or 1.0)
            calc_leverage = getattr(executor, "calculate_dynamic_leverage", None)
            if callable(calc_leverage):
                try:
                    effective_leverage = float(
                        calc_leverage(
                            float(signal.get("confidence", 0.0) or 0.0),
                            float(signal.get("score", 0.0) or 0.0),
                            atr_pct=signal.get("atr_pct"),
                        )
                    )
                except Exception:
                    pass
            notional = trade_usdt * effective_leverage
            gross_pnl = (
                notional * (current_price / entry - 1.0)
                if signal_action == "BUY"
                else notional * (1.0 - current_price / entry)
            )
            estimated_fees = notional * float(getattr(executor, "fee_rate", 0.0) or 0.0) * 2.0
            net_pnl = gross_pnl - estimated_fees
            pnl_estimate = {
                "active": False,
                "side": signal_action,
                "entry": entry,
                "price": current_price,
                "trade_usdt": trade_usdt,
                "effective_leverage": effective_leverage,
                "notional_usdt": notional,
                "gross_pnl": gross_pnl,
                "gross_pnl_pct": (gross_pnl / notional * 100.0) if notional > 0 else 0.0,
                "estimated_fees": estimated_fees,
                "net_pnl": net_pnl,
                "net_pnl_pct": (net_pnl / notional * 100.0) if notional > 0 else 0.0,
                "formula": "LONG: notional × (price / entry - 1); SHORT: notional × (1 - price / entry)",
            }

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
        "positions": normalized_positions,
        "pending_entry": pending_entry,
        "pending_exit": pending_exit,
        "open_orders": copy.deepcopy(open_orders or []),
        "closed_trades": copy.deepcopy(closed_trades),
        "status_lines": list(status_lines or []),
        "pivot_data": copy.deepcopy(pivot_data or {}),
        "mtf_context": copy.deepcopy(mtf_context or {}),
        "latest_indicators": copy.deepcopy(latest_indicators or {}),
        "ai_overlay": copy.deepcopy(ai_overlay_state or {}),
        "pnl_estimate": pnl_estimate,
        "chart": list(chart_bars or []),
        "config": {
            "execution": copy.deepcopy(cfg.get("execution", {}) or {}),
            "strategy": copy.deepcopy(cfg.get("strategy", {}) or {}),
            "mtf": copy.deepcopy(cfg.get("mtf", {}) or {}),
        }
    }
