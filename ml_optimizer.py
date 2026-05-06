import json
import logging
import os
import re
import threading
from collections import deque
from datetime import UTC, datetime

import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.json")
LEARNING_STATE_FILE = os.path.join(os.path.dirname(__file__), "learning_state.json")

ACTIVE_FEATURES = ["mr", "vwap", "bb", "macd", "pa", "smc", "sr", "kdj", "st"]
DEFAULT_WEIGHTS = {
    "mr": 0.10,
    "vwap": 0.10,
    "bb": 0.05,
    "macd": 0.05,
    "pa": 0.15,
    "smc": 0.15,
    "sr": 0.10,
    "kdj": 0.10,
    "st": 0.10,
}

logger = logging.getLogger(__name__)


class SessionTracker:
    """Collects completed trades in-memory for the current session only.

    On each bot restart the session starts fresh — no stale historical data
    is carried over.  The tracker stores the entry-side indicator scores
    (the ``reason`` string from the signal) alongside PnL so the ML
    optimiser can learn *within* the session.
    """

    def __init__(self, max_trades: int = 1000):
        self._trades: deque[dict] = deque(maxlen=max_trades)
        self._lock = threading.Lock()
        self._pending_entries: dict[int, dict] = {}

    def register_entry(self, trade_id: int, side: str, reason: str):
        with self._lock:
            self._pending_entries[trade_id] = {
                "side": side.upper(),
                "reason": reason or "",
            }

    def register_exit(self, trade_id: int, net_pnl: float, entry_price: float = 0.0, exit_price: float = 0.0):
        with self._lock:
            entry_info = self._pending_entries.pop(trade_id, None)
            if entry_info is None:
                return
            self._trades.append({
                "side": entry_info["side"],
                "reason": entry_info["reason"],
                "net_pnl": float(net_pnl),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "profitable": 1 if float(net_pnl) > 0 else 0,
                "timestamp": datetime.now(UTC).isoformat(),
            })

    def get_trades(self) -> list[dict]:
        with self._lock:
            return list(self._trades)

    def clear(self):
        with self._lock:
            self._trades.clear()
            self._pending_entries.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._trades)


_session_tracker = SessionTracker()


def get_session_tracker() -> SessionTracker:
    return _session_tracker


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {}
    for key in ACTIVE_FEATURES:
        try:
            cleaned[key] = max(0.0, float(weights.get(key, 0.0)))
        except (TypeError, ValueError):
            cleaned[key] = 0.0

    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: round(value / total, 4) for key, value in cleaned.items()}


def parse_reason(reason_str: str) -> dict[str, float]:
    scores = {key: 0.0 for key in ACTIVE_FEATURES}
    if not reason_str:
        return scores

    try:
        for raw_key, raw_value in re.findall(r"([A-Za-z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?)", str(reason_str)):
            key = raw_key.lower()
            if key == "ob":
                key = "smc"
            if key in scores:
                scores[key] = float(raw_value)
    except Exception:
        return scores
    return scores


def _build_session_dataset(trades: list[dict]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows = []
    profitable = []
    pnl_list = []
    for trade in trades:
        scores = parse_reason(str(trade.get("reason", "")))
        side = str(trade.get("side", "")).upper()
        direction = -1.0 if side == "SELL" else 1.0
        rows.append({key: scores[key] * direction for key in ACTIVE_FEATURES})
        profitable.append(trade.get("profitable", 1 if trade.get("net_pnl", 0) > 0 else 0))
        pnl_list.append(float(trade.get("net_pnl", 0.0)))

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    X = pd.DataFrame(rows, columns=ACTIVE_FEATURES).fillna(0.0)
    y = pd.Series(profitable, dtype=int)
    completed = pd.DataFrame({
        "Side": [t.get("side", "") for t in trades],
        "net_pnl": pnl_list,
        "profitable": profitable,
    })
    return X, y, completed


def _apply_ai_advisor(
    state: dict,
    weights: dict[str, float],
    ai_model: str,
    max_weight_shift: float = 0.12,
) -> tuple[dict[str, float], dict]:
    """Ask an LLM for a bounded risk review of the learned weights.

    Uses the HybridAIOrchestrator (DeepSeek / Gemini / OpenRouter / OpenAI
    fallback chain) instead of calling OpenAI directly.
    """
    advisor_state = {
        "enabled": False,
        "applied": False,
        "model": ai_model,
        "rationale": "",
        "error": "",
    }

    try:
        from agents import HybridAIOrchestrator

        orch = HybridAIOrchestrator(model=ai_model)
        if not orch.enabled:
            advisor_state["error"] = "No AI providers available"
            return weights, advisor_state

        advisor_state["enabled"] = True
        max_weight_shift = max(0.0, min(0.30, float(max_weight_shift)))
        lower = round(1.0 - max_weight_shift, 4)
        upper = round(1.0 + max_weight_shift, 4)

        payload = {
            "stats": {
                "completed_trades": state.get("completed_trades"),
                "win_rate": state.get("win_rate"),
                "net_pnl": state.get("net_pnl"),
                "avg_win": state.get("avg_win"),
                "avg_loss": state.get("avg_loss"),
                "risk_multiplier": state.get("risk_multiplier"),
                "model_type": state.get("model_type"),
                "train_accuracy": state.get("train_accuracy"),
                "holdout_accuracy": state.get("holdout_accuracy"),
            },
            "weights": weights,
            "allowed_weight_multiplier_range": [lower, upper],
            "features": ACTIVE_FEATURES,
        }

        system_prompt = (
            "Return valid JSON only with keys: weight_multipliers, risk_multiplier, rationale. "
            f"weight_multipliers must include only these keys: { ACTIVE_FEATURES }. "
            f"Each multiplier must be between { lower } and { upper }. "
            "risk_multiplier must be between 0.10 and 1.00."
        )

        raw = orch._call_llm(
            system_prompt,
            json.dumps(payload, ensure_ascii=False),
            max_tokens=300,
            json_mode=True,
        )

        from agents import _extract_json_object

        data = _extract_json_object(raw)

        multipliers = data.get("weight_multipliers", {}) or {}
        adjusted = {}
        for key in ACTIVE_FEATURES:
            try:
                mult = float(multipliers.get(key, 1.0))
            except (TypeError, ValueError):
                mult = 1.0
            mult = max(lower, min(upper, mult))
            adjusted[key] = float(weights.get(key, 0.0)) * mult

        adjusted = _normalize_weights(adjusted)

        base_risk = float(state.get("risk_multiplier", 1.0) or 1.0)
        try:
            ai_risk = float(data.get("risk_multiplier", base_risk))
        except (TypeError, ValueError):
            ai_risk = base_risk
        ai_risk = max(0.10, min(1.0, ai_risk))

        stats_are_positive = float(state.get("net_pnl", 0.0) or 0.0) > 0 and float(state.get("win_rate", 0.0) or 0.0) >= 0.50
        state["risk_multiplier"] = ai_risk if stats_are_positive else min(base_risk, ai_risk)

        advisor_state.update({
            "applied": True,
            "rationale": str(data.get("rationale", ""))[:300],
            "risk_multiplier": state["risk_multiplier"],
        })
        return adjusted, advisor_state
    except Exception as e:
        advisor_state["error"] = str(e)[:300]
        logger.warning(f"AI advisor failed: {e}")
        return weights, advisor_state


def optimize_weights(
    min_trades: int = 10,
    shrinkage: float = 0.35,
    ai_enabled: bool = False,
    ai_model: str = "deepseek-chat",
    ai_max_weight_shift: float = 0.12,
    quiet: bool = False,
    session_trades: list[dict] | None = None,
) -> dict | None:
    """Learn indicator weights from session trades only.

    Uses *only* trades collected during the current bot session — no
    historical CSV data is read.  This ensures the model adapts to
    prevailing market conditions and immediately penalises indicators
    that are producing losses *right now*.

    When ``session_trades`` is ``None``, falls back to the global
    :class:`SessionTracker`.
    """
    tracker = get_session_tracker()
    trades = session_trades if session_trades is not None else tracker.get_trades()

    if not trades:
        if not quiet:
            logger.info("No session trades available for learning.")
        return None

    X, y, completed = _build_session_dataset(trades)
    if X.empty or len(completed) < int(min_trades):
        if not quiet:
            logger.info(f"Not enough session trades to optimise weights (found {len(completed)}, need {min_trades}).")
        return None

    if len(set(y.tolist())) < 2:
        if not quiet:
            logger.info("Model cannot be trained yet: need both wins and losses in the session.")
        return None

    train_accuracy = None
    holdout_accuracy = None

    if SKLEARN_AVAILABLE:
        split_idx = int(len(X) * 0.8)
        use_holdout = len(X) >= 20 and split_idx > 0 and split_idx < len(X)

        scaler = StandardScaler()
        if use_holdout:
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = scaler.fit_transform(X)
            y_train = y
            X_test_scaled = None
            y_test = None

        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train_scaled, y_train)

        raw_weights = {}
        for key, coef in zip(ACTIVE_FEATURES, model.coef_[0]):
            raw_weights[key] = max(0.0, float(coef))

        train_accuracy = float(accuracy_score(y_train, model.predict(X_train_scaled)))
        if use_holdout and X_test_scaled is not None and y_test is not None:
            holdout_accuracy = float(accuracy_score(y_test, model.predict(X_test_scaled)))
            state_model_type = "logistic_regression_session_holdout"
        else:
            state_model_type = "logistic_regression_session"
    else:
        outcome = (y.astype(float) * 2.0) - 1.0
        raw_weights = {}
        for key in ACTIVE_FEATURES:
            edge = float((X[key].astype(float) * outcome).mean())
            raw_weights[key] = max(0.0, edge)
        state_model_type = "correlation_fallback_session"

    learned = _normalize_weights(raw_weights)
    shrinkage = max(0.0, min(1.0, float(shrinkage)))

    # Adaptive shrinkage: use a higher learning ratio earlier in the session
    # so corrections take effect fast, then taper off as sample grows.
    n = len(completed)
    session_boost = max(0.0, 0.20 - (n / 200.0) * 0.20)
    effective_shrinkage = min(1.0, shrinkage + session_boost)

    # Holdout gate: discard learned weights if model can't generalise
    if holdout_accuracy is not None and holdout_accuracy < 0.50:
        if not quiet:
            logger.warning(f"Holdout accuracy {holdout_accuracy:.1%} < 50%% — discarding learned weights.")
        learned = DEFAULT_WEIGHTS.copy()
        effective_shrinkage = 0.0
    elif holdout_accuracy is not None and holdout_accuracy < 0.55:
        if not quiet:
            logger.info(f"Holdout accuracy {holdout_accuracy:.1%} weak — reducing learned weight influence.")
        effective_shrinkage = min(effective_shrinkage + 0.15, 0.35)

    blended = _normalize_weights({
        key: (DEFAULT_WEIGHTS[key] * (1.0 - effective_shrinkage)) + (learned[key] * effective_shrinkage)
        for key in ACTIVE_FEATURES
    })

    net_pnl_values = pd.to_numeric(completed["net_pnl"], errors="coerce").fillna(0.0)
    wins = net_pnl_values[net_pnl_values > 0]
    losses = net_pnl_values[net_pnl_values <= 0]
    win_rate = float((net_pnl_values > 0).mean())
    total_net_pnl = float(net_pnl_values.sum())

    # Aggressive risk reduction during losing streaks within the session
    consecutive_losses = 0
    for pnl in reversed(net_pnl_values.tolist()):
        if pnl <= 0:
            consecutive_losses += 1
        else:
            break

    if total_net_pnl < 0 and win_rate < 0.35:
        risk_multiplier = 0.10
    elif consecutive_losses >= 5:
        risk_multiplier = 0.25
    elif consecutive_losses >= 3:
        risk_multiplier = 0.40
    elif total_net_pnl < 0 and win_rate < 0.45:
        risk_multiplier = 0.25
    elif total_net_pnl < 0 or win_rate < 0.50:
        risk_multiplier = 0.50
    elif win_rate < 0.55:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 1.0

    state_model_type_final = state_model_type
    state = {
        "learned_at": datetime.now(UTC).isoformat(),
        "source": "session",
        "model_type": state_model_type_final,
        "completed_trades": int(len(completed)),
        "win_rate": round(win_rate, 4),
        "net_pnl": round(total_net_pnl, 6),
        "avg_win": round(float(wins.mean()) if not wins.empty else 0.0, 6),
        "avg_loss": round(float(losses.mean()) if not losses.empty else 0.0, 6),
        "consecutive_losses": consecutive_losses,
        "risk_multiplier": risk_multiplier,
        "train_accuracy": round(train_accuracy, 4) if train_accuracy is not None else None,
        "holdout_accuracy": round(holdout_accuracy, 4) if holdout_accuracy is not None else None,
        "shrinkage_used": round(effective_shrinkage, 4),
        "weights": blended,
    }

    if ai_enabled:
        blended, advisor_state = _apply_ai_advisor(
            state,
            blended,
            ai_model=ai_model,
            max_weight_shift=ai_max_weight_shift,
        )
        state["weights"] = blended
        state["ai_advisor"] = advisor_state
    else:
        state["ai_advisor"] = {"enabled": False, "applied": False}

    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(blended, f, indent=4)
    with open(LEARNING_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    if not quiet:
        logger.info(f"Session learning updated: {len(completed)} trades, WR {win_rate:.1%}, risk {risk_multiplier:.2f}x, PnL {total_net_pnl:+.4f}")
    return state


if __name__ == "__main__":
    optimize_weights()