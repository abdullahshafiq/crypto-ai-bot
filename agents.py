import os
import logging
import json
import re
import time
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def _call_deepseek(
    system_prompt: str,
    user_content: str,
    api_key: str,
    model: str = _DEEPSEEK_DEFAULT_MODEL,
    max_tokens: int = 250,
    json_mode: bool = False,
) -> str:
    client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**params)
    except Exception:
        if not json_mode:
            raise
        params.pop("response_format", None)
        response = client.chat.completions.create(**params)
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("DeepSeek returned empty response (rate limit or content filter)")
    return content


def _extract_json_object(raw: str) -> dict:
    """
    Accept strict JSON, fenced JSON, or prose around one JSON object.
    Raises ValueError with a short raw preview if parsing still fails.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    payload = text[start:end + 1] if start != -1 and end != -1 and end > start else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        preview = text.replace("\n", " ")[:300]
        raise ValueError(f"Invalid JSON from AI: {exc}; raw={preview!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"AI JSON payload was not an object: {type(data).__name__}")
    return data


def _call_gemini(system_prompt: str, user_content: str, max_tokens: int = 250) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{
                "text": f"System Instruction: {system_prompt}\n\nUser Content: {user_content}"
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.0
        }
    }
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
    if r.status_code == 200:
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
    raise RuntimeError(f"Gemini API returned status {r.status_code}: {r.text}")


def _call_openrouter(system_prompt: str, user_content: str, max_tokens: int = 250) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")
    url = "https://openrouter.ai/api/v1/chat/completions"

    free_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "liquid/lfm-2.5-1.2b-thinking:free"
    ]

    errors = []
    for model in free_models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        try:
            r = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://crypto-ai-bot.local",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    pass
            errors.append(f"Model {model} failed with status {r.status_code}: {r.text}")
        except Exception as e:
            errors.append(f"Model {model} threw exception: {e}")

    raise RuntimeError(f"All free OpenRouter models failed. Errors: {'; '.join(errors)}")



class HybridAIOrchestrator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.last_macro_regime = "NEUTRAL"
        self._last_trade_eval = None
        self._last_overlay = None
        self._last_trade_time = 0.0
        self._last_overlay_time = 0.0
        self._last_regime_time = 0.0

        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", _DEEPSEEK_DEFAULT_MODEL)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        self.enabled = bool(
            self.gemini_key
            or self.deepseek_key
            or self.openrouter_key
            or self.openai_key
        )

        if self.enabled:
            if self.openai_key:
                self.client = OpenAI(api_key=self.openai_key)
            else:
                self.client = None
            providers = []
            if self.gemini_key:
                providers.append("Gemini")
            if self.deepseek_key:
                providers.append(f"DeepSeek({self.deepseek_model})")
            if self.openrouter_key:
                providers.append("OpenRouter")
            if self.openai_key:
                providers.append("OpenAI")
            logger.info(f"AI Orchestrator initialized ({', '.join(providers)}).")
        else:
            self.client = None
            logger.warning("No AI API key configured. AI override disabled (defaulting to NEUTRAL).")

    def _call_llm(self, system_prompt: str, user_content: str, max_tokens: int = 250, json_mode: bool = False) -> str:
        # 1) Gemini direct API (fast, free tier)
        if self.gemini_key:
            try:
                return _call_gemini(system_prompt, user_content, max_tokens)
            except Exception as e:
                logger.warning(f"Gemini call failed ({e}). Falling back...")

        # 2) DeepSeek (cheap, OpenAI-compatible, $0.14/M input)
        if self.deepseek_key:
            try:
                return _call_deepseek(
                    system_prompt,
                    user_content,
                    api_key=self.deepseek_key,
                    model=self.deepseek_model,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except Exception as e:
                logger.warning(f"DeepSeek call failed ({e}). Falling back...")

        # 3) OpenRouter free models
        if self.openrouter_key:
            try:
                return _call_openrouter(system_prompt, user_content, max_tokens)
            except Exception as e:
                logger.warning(f"OpenRouter fallback failed ({e}). Trying next...")

        # 4) OpenAI (most expensive, last resort)
        if self.client and self.openai_key:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                logger.error(f"OpenAI final fallback failed: {e}")
                raise e

        raise RuntimeError("No available AI providers succeeded.")

    def evaluate_trade(self, context: dict, model: str | None = None) -> dict:
        fallback = {"decision": "ALLOW", "hold_minutes": 0, "confidence": 0.0, "rationale": "AI disabled"}
        if not self.enabled:
            return fallback

        now = time.time()
        if now - getattr(self, "_last_trade_time", 0.0) < 60:
            if getattr(self, "_last_trade_eval", None):
                return self._last_trade_eval
            return fallback
        self._last_trade_time = now

        system_prompt = (
            "Scalp hunter: default ALLOW. "
            "If 3m/5m show momentum, ALLOW immediately. "
            "hold_minutes must be 0. Trade NOW or don't trade. "
            "JSON only: {decision, confidence, rationale}."
        )

        user_content = json.dumps(context, ensure_ascii=False)

        try:
            raw = self._call_llm(system_prompt, user_content, max_tokens=250, json_mode=True)
            data = _extract_json_object(raw)

            decision = str(data.get("decision", "ALLOW")).upper()
            if decision not in {"ALLOW", "VETO"}:
                decision = "ALLOW"

            hold_minutes = int(data.get("hold_minutes", 0) or 0)
            if hold_minutes < 0:
                hold_minutes = 0

            confidence = float(data.get("confidence", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))

            rationale = str(data.get("rationale", "") or "")[:240]

            out = {
                "decision": decision,
                "hold_minutes": hold_minutes,
                "confidence": confidence,
                "rationale": rationale,
            }
            self._last_trade_eval = out
            return out

        except Exception as e:
            logger.error(f"AI trade evaluation failed: {e}")
            return {"decision": "ALLOW", "hold_minutes": 0, "confidence": 0.0, "rationale": f"AI error: {e}"}

    def evaluate_overlay(self, context: dict, model: str | None = None) -> dict:
        """
        Hybrid overlay: rules-based for clear signals, AI tiebreaker when MTF is confused.
        """
        mtf = context.get("mtf", {}) or {}
        primary_tfs = ["3m", "5m", "15m"]
        bull_votes = sum(1 for t in primary_tfs if str(mtf.get(t, "")).upper() == "BULL")
        bear_votes = sum(1 for t in primary_tfs if str(mtf.get(t, "")).upper() == "BEAR")

        # Clear signal: 2+ timeframes agree → rules-based, zero tokens
        is_clear = bull_votes >= 2 or bear_votes >= 2

        rsi = context.get("indicators", {}).get("rsi")
        rsi_val = float(rsi) if rsi is not None else 50.0
        rsi_extreme = rsi_val > 75 or rsi_val < 25

        # Use AI tiebreaker only when signals are confused
        use_ai = not is_clear or rsi_extreme

        if not use_ai or not self.enabled:
            # Rules-based path (zero tokens, fast)
            if bull_votes >= 2:
                bias, risk_mode = "LONG_ONLY", "NORMAL"
                rationale = f"MTF: {bull_votes}/3 bullish"
            elif bear_votes >= 2:
                bias, risk_mode = "SHORT_ONLY", "NORMAL"
                rationale = f"MTF: {bear_votes}/3 bearish"
            else:
                bias, risk_mode = "NEUTRAL", "CAUTIOUS"
                rationale = f"MTF mixed ({bull_votes}B/{bear_votes}S)"

            if rsi_extreme:
                risk_mode = "RISK_OFF"
                rationale += " | RSI extreme"

            out = {
                "bias": bias, "risk_mode": risk_mode,
                "entry_style": "BREAKOUTS",
                "avoid_new_entries": risk_mode == "RISK_OFF",
                "max_hold_minutes": 0,
                "confidence": max(bull_votes, bear_votes) / 3.0,
                "rationale": rationale,
            }
            self._last_overlay = out
            self._last_overlay_time = time.time()
            return out

        # Confused signal — AI tiebreaker
        now = time.time()
        if now - getattr(self, "_last_overlay_time", 0.0) < 60:
            if getattr(self, "_last_overlay", None):
                return self._last_overlay
        self._last_overlay_time = now

        try:
            system_prompt = (
                "Tactical scalp bias from 1m/3m/5m/10m/15m context; primary confirmation is 3m/5m/15m. "
                "Favor momentum trends. "
                "JSON only: {bias, risk_mode, entry_style, avoid_new_entries, max_hold_minutes, confidence, rationale}. "
                "bias: LONG_ONLY|SHORT_ONLY|NEUTRAL. risk_mode: NORMAL|CAUTIOUS|RISK_OFF. "
                "entry_style: BUY_PULLBACKS|SELL_RALLIES|BREAKOUTS|MIXED. confidence: 0..1. "
                "avoid_new_entries may be true only when risk_mode is RISK_OFF due to extreme risk; "
                "for normal neutral or choppy markets use CAUTIOUS with avoid_new_entries false."
            )
            raw = self._call_llm(system_prompt, json.dumps(context, ensure_ascii=False), max_tokens=300, json_mode=True)
            data = _extract_json_object(raw)

            bias = str(data.get("bias", "NEUTRAL")).upper()
            if bias not in {"LONG_ONLY", "SHORT_ONLY", "NEUTRAL"}:
                bias = "NEUTRAL"
            risk_mode = str(data.get("risk_mode", "NORMAL")).upper()
            if risk_mode not in {"NORMAL", "CAUTIOUS", "RISK_OFF"}:
                risk_mode = "NORMAL"
            entry_style = str(data.get("entry_style", "MIXED")).upper()
            if entry_style not in {"BUY_PULLBACKS", "SELL_RALLIES", "BREAKOUTS", "MIXED"}:
                entry_style = "MIXED"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5) or 0.5)))
            avoid_new_entries = risk_mode == "RISK_OFF"

            out = {
                "bias": bias, "risk_mode": risk_mode,
                "entry_style": entry_style,
                "avoid_new_entries": avoid_new_entries,
                "max_hold_minutes": max(0, int(data.get("max_hold_minutes", 0) or 0)),
                "confidence": confidence,
                "rationale": str(data.get("rationale", "AI tiebreaker") or "")[:180],
            }
            self._last_overlay = out
            return out
        except Exception as e:
            logger.warning(f"AI overlay tiebreaker failed: {e}")
            # Smart fallback: even one vote beats NEUTRAL
            if bear_votes > bull_votes:
                bias, conf = "SHORT_ONLY", bear_votes / 3.0
            elif bull_votes > bear_votes:
                bias, conf = "LONG_ONLY", bull_votes / 3.0
            else:
                bias, conf = "NEUTRAL", 0.0
            return {
                "bias": bias, "risk_mode": "CAUTIOUS",
                "entry_style": "MIXED", "avoid_new_entries": False,
                "max_hold_minutes": 0, "confidence": conf,
                "rationale": f"Mixed MTF ({bull_votes}B/{bear_votes}S), AI parse failed - leaning {bias}",
            }

    def determine_macro_regime(self, news_headlines: list, quant_context: str = "") -> str:
        """
        Rule-based regime — zero tokens. Uses ADX, MACD, price vs EMA from quant context.
        """

        # Parse quant_context: "Price: X\nRSI: Y\nADX: Z\nFutures Funding Rate: W%"
        adx = 0.0
        rsi = 50.0
        price = 0.0
        funding = 0.0

        text_context = str(quant_context)
        adx_m = re.search(r"ADX(?:\s*\([^)]*\))?\s*[:=]\s*(-?[\d.]+)", text_context, flags=re.IGNORECASE)
        rsi_m = re.search(r"RSI(?:\s*\([^)]*\))?\s*[:=]\s*(-?[\d.]+)", text_context, flags=re.IGNORECASE)
        price_m = re.search(r"Price\s*[:=]\s*(-?[\d.]+)", text_context, flags=re.IGNORECASE)
        fund_m = re.search(r"Funding[^:=]*[:=]\s*(-?[\d.]+)\s*(%)?", text_context, flags=re.IGNORECASE)

        if adx_m:
            adx = float(adx_m.group(1))
        if rsi_m:
            rsi = float(rsi_m.group(1))
        if price_m:
            price = float(price_m.group(1))
        if fund_m:
            funding = float(fund_m.group(1))
            if fund_m.group(2):
                funding /= 100.0

        # No data? return last known
        if adx == 0.0 and rsi == 50.0:
            return getattr(self, "last_macro_regime", "NEUTRAL")

        now = time.time()
        if now - getattr(self, "_last_regime_time", 0.0) < 60:
            return getattr(self, "last_macro_regime", "NEUTRAL")
        self._last_regime_time = now

        # Quantitative regime rules
        trend_strong = adx >= 25
        trend_weak = adx < 18
        overbought = rsi > 70
        oversold = rsi < 30
        funding_extreme = abs(funding) > 0.05

        if funding_extreme or (trend_strong and overbought and rsi > 80):
            regime = "VOLATILE"
        elif trend_strong and not overbought and rsi >= 50:
            regime = "BULLISH"
        elif trend_strong and not oversold and rsi < 50:
            regime = "BEARISH"
        elif trend_weak:
            regime = "NEUTRAL"
        else:
            # Mid ADX, mid RSI — check RSI bias
            regime = "BULLISH" if rsi >= 55 else ("BEARISH" if rsi <= 45 else "NEUTRAL")

        self.last_macro_regime = regime
        logger.info(f"Regime: {regime} (ADX:{adx:.0f} RSI:{rsi:.0f} Funding:{funding:.2%})")
        return regime
