import json
import os
import re
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

TRADE_LOG_V2 = os.path.join(os.path.dirname(__file__), "trade_log_futures.csv")
TRADE_LOG = os.path.join(os.path.dirname(__file__), "trade_log.csv")
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
    """Extract indicator scores from the signal reason string."""
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


def _load_completed_trades(log_path: str) -> pd.DataFrame:
    df = pd.read_csv(log_path)

    if {"trade_id", "event", "price"}.issubset(df.columns):
        df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce")
        df = df.dropna(subset=["trade_id"]).copy()
        df["trade_id"] = df["trade_id"].astype(int)
        if "timestamp" in df.columns:
            df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("_ts")

        open_entries: dict[int, list[pd.Series]] = {}
        completed_rows = []
        for _, row in df.iterrows():
            event = str(row.get("event", "")).upper()
            trade_id = int(row.get("trade_id"))
            if event == "ENTRY":
                open_entries.setdefault(trade_id, []).append(row)
            elif event == "EXIT" and open_entries.get(trade_id):
                entry = open_entries[trade_id].pop(0)
                entry_fee = pd.to_numeric(entry.get("fees", 0.0), errors="coerce")
                exit_fee = pd.to_numeric(row.get("fees", 0.0), errors="coerce")
                pnl = pd.to_numeric(row.get("pnl", 0.0), errors="coerce")
                entry_fee = 0.0 if pd.isna(entry_fee) else float(entry_fee)
                exit_fee = 0.0 if pd.isna(exit_fee) else float(exit_fee)
                pnl = 0.0 if pd.isna(pnl) else float(pnl)
                completed_rows.append({
                    "trade_id": trade_id,
                    "Entry_Price": entry.get("price"),
                    "Exit_Price": row.get("price"),
                    "Reason": entry.get("reason", ""),
                    "Side": entry.get("side", ""),
                    "timestamp_exit": row.get("timestamp"),
                    "net_pnl": pnl - entry_fee - exit_fee,
                })

        if not completed_rows:
            return pd.DataFrame()

        completed = pd.DataFrame(completed_rows)
        completed["Entry_Price"] = pd.to_numeric(completed["Entry_Price"], errors="coerce")
        completed["Exit_Price"] = pd.to_numeric(completed["Exit_Price"], errors="coerce")
        completed["timestamp_exit"] = pd.to_datetime(completed["timestamp_exit"], errors="coerce")
        completed["profitable"] = (completed["net_pnl"] > 0).astype(int)
        completed = completed.dropna(subset=["Entry_Price", "Exit_Price"]).copy()
        return completed.sort_values("timestamp_exit")

    if not {"Entry_Price", "Exit_Price", "Reason"}.issubset(df.columns):
        return pd.DataFrame()

    completed = df[df["Exit_Price"].notna()].copy()
    pnl_col = "PnL" if "PnL" in completed.columns else "pnl"
    if pnl_col in completed.columns:
        completed["net_pnl"] = pd.to_numeric(completed[pnl_col], errors="coerce").fillna(0.0)
    else:
        completed["net_pnl"] = pd.to_numeric(completed["Exit_Price"], errors="coerce") - pd.to_numeric(completed["Entry_Price"], errors="coerce")
    completed["Side"] = completed.get("Side", "BUY")
    completed["profitable"] = (completed["net_pnl"] > 0).astype(int)
    return completed


def build_learning_dataset(max_recent_trades: int = 300) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    log_path = TRADE_LOG_V2 if os.path.exists(TRADE_LOG_V2) else TRADE_LOG
    if not os.path.exists(log_path):
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    completed = _load_completed_trades(log_path)
    if completed.empty:
        return pd.DataFrame(), pd.Series(dtype=int), completed

    if max_recent_trades > 0:
        completed = completed.tail(int(max_recent_trades)).copy()

    rows = []
    for _, row in completed.iterrows():
        scores = parse_reason(str(row.get("Reason", "")))
        side = str(row.get("Side", "")).upper()
        direction = -1.0 if side == "SELL" else 1.0
        # Align each indicator with the chosen trade direction. A positive value
        # means the indicator agreed with the entry side; negative means conflict.
        rows.append({key: scores[key] * direction for key in ACTIVE_FEATURES})

    X = pd.DataFrame(rows, columns=ACTIVE_FEATURES).fillna(0.0)
    y = completed["profitable"].astype(int)
    return X, y, completed


def _apply_ai_advisor(
    state: dict,
    weights: dict[str, float],
    ai_model: str,
    max_weight_shift: float = 0.12,
) -> tuple[dict[str, float], dict]:
    """
    Ask OpenAI for a bounded risk review of the learned weights.

    The AI is not allowed to create trades or directly raise risk after poor
    results. It can only make small weight nudges and recommend equal/lower
    sizing unless the statistical sample is already positive.
    """
    advisor_state = {
        "enabled": False,
        "applied": False,
        "model": ai_model,
        "rationale": "",
        "error": "",
    }

    try:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            advisor_state["error"] = "OPENAI_API_KEY missing"
            return weights, advisor_state

        from openai import OpenAI

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
            "You are a conservative model-risk advisor for a crypto scalping bot. "
            "You do not place trades. You only review learned indicator weights and risk sizing. "
            "Return valid JSON only with keys: weight_multipliers, risk_multiplier, rationale. "
            f"weight_multipliers must include only these keys: {ACTIVE_FEATURES}. "
            f"Each multiplier must be between {lower} and {upper}. "
            "risk_multiplier must be between 0.10 and 1.00. "
            "If net_pnl is negative or win_rate is below 0.50, do not increase risk. "
            "Prefer reducing noisy/overfit indicators and explain briefly."
        )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = (response.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start:end + 1] if start != -1 and end != -1 and end > start else raw)

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
        return weights, advisor_state


def optimize_weights(
    min_trades: int = 30,
    max_recent_trades: int = 300,
    shrinkage: float = 0.35,
    ai_enabled: bool = False,
    ai_model: str = "gpt-4o-mini",
    ai_max_weight_shift: float = 0.12,
    quiet: bool = False,
) -> dict | None:
    """
    Learn indicator weights from completed trades.

    The learned model is intentionally conservative: it only increases weights
    for indicators that historically agreed with profitable entries, then blends
    them back toward the default weights to avoid overfitting a small sample.
    """
    X, y, completed = build_learning_dataset(max_recent_trades=max_recent_trades)
    if completed.empty:
        if not quiet:
            print("No completed trades found.")
        return None

    if len(completed) < int(min_trades):
        if not quiet:
            print(f"Not enough completed trades to optimize weights (found {len(completed)}, need {min_trades}).")
        return None

    if len(set(y.tolist())) < 2:
        if not quiet:
            print("Model cannot be trained yet: need both wins and losses.")
        return None

    train_accuracy = None
    holdout_accuracy = None

    if SKLEARN_AVAILABLE:
        split_idx = int(len(X) * 0.8)
        use_holdout = len(X) >= 50 and split_idx > 0 and split_idx < len(X)

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
        model_type = "logistic_regression"
    else:
        # Fallback learner: reward indicators whose agreement with the entry
        # direction is associated with wins, penalize those associated with losses.
        outcome = (y.astype(float) * 2.0) - 1.0
        raw_weights = {}
        for key in ACTIVE_FEATURES:
            edge = float((X[key].astype(float) * outcome).mean())
            raw_weights[key] = max(0.0, edge)
        model_type = "correlation_fallback"

    learned = _normalize_weights(raw_weights)
    shrinkage = max(0.0, min(1.0, float(shrinkage)))
    blended = _normalize_weights({
        key: (DEFAULT_WEIGHTS[key] * (1.0 - shrinkage)) + (learned[key] * shrinkage)
        for key in ACTIVE_FEATURES
    })

    net_pnl = pd.to_numeric(completed["net_pnl"], errors="coerce").fillna(0.0)
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl <= 0]
    win_rate = float((net_pnl > 0).mean())
    total_net_pnl = float(net_pnl.sum())
    if total_net_pnl < 0 and win_rate < 0.45:
        risk_multiplier = 0.25
    elif total_net_pnl < 0 or win_rate < 0.50:
        risk_multiplier = 0.50
    elif win_rate < 0.55:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 1.0
    state = {
        "learned_at": datetime.now(UTC).isoformat(),
        "model_type": model_type,
        "completed_trades": int(len(completed)),
        "win_rate": round(win_rate, 4),
        "net_pnl": round(total_net_pnl, 6),
        "avg_win": round(float(wins.mean()) if not wins.empty else 0.0, 6),
        "avg_loss": round(float(losses.mean()) if not losses.empty else 0.0, 6),
        "risk_multiplier": risk_multiplier,
        "train_accuracy": round(train_accuracy, 4) if train_accuracy is not None else None,
        "holdout_accuracy": round(holdout_accuracy, 4) if holdout_accuracy is not None else None,
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
        print(f"Saved learned weights to {WEIGHTS_FILE}")
        print(json.dumps(state, indent=2))
    return state


if __name__ == "__main__":
    optimize_weights()
