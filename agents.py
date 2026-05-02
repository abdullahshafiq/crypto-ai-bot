import os
import logging
import json
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


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
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        self.enabled = bool(self.gemini_key or self.openrouter_key or self.openai_key)

        if self.enabled:
            if self.openai_key:
                self.client = OpenAI(api_key=self.openai_key)
            else:
                self.client = None
            logger.info(f"AI Orchestrator initialized (Gemini/OpenRouter/OpenAI enabled).")
        else:
            self.client = None
            logger.warning("No AI API key configured. AI override disabled (defaulting to NEUTRAL).")

    def _call_llm(self, system_prompt: str, user_content: str, max_tokens: int = 250) -> str:
        # First try Gemini direct API
        if self.gemini_key:
            try:
                return _call_gemini(system_prompt, user_content, max_tokens)
            except Exception as e:
                logger.warning(f"Gemini Direct Call failed ({e}). Attempting OpenRouter fallback...")

        # Fallback to OpenRouter
        if self.openrouter_key:
            try:
                return _call_openrouter(system_prompt, user_content, max_tokens)
            except Exception as e:
                logger.warning(f"OpenRouter fallback failed ({e}). Trying OpenAI fallback if available...")

        # Final fallback to OpenAI if client is available
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

        import time
        now = time.time()
        if now - getattr(self, "_last_trade_time", 0.0) < 60:
            if getattr(self, "_last_trade_eval", None):
                return self._last_trade_eval
            return fallback
        self._last_trade_time = now

        system_prompt = (
            "You are a 'Bias Toward Action' scalp hunter. "
            "Your default posture is to ALLOW trades. "
            "If the 3m or 5m charts show ANY sign of momentum, ALLOW the trade immediately. "
            "IMPORTANT: DO NOT tell the bot to wait. 'hold_minutes' must ALWAYS be 0. "
            "We trade NOW or we don't trade at all. "
            "Output valid JSON ONLY: {decision, confidence, rationale}."
        )

        user_content = json.dumps(context, ensure_ascii=False)

        try:
            raw = self._call_llm(system_prompt, user_content, max_tokens=200)
            # Be tolerant to accidental prose: extract the first JSON object.
            start = raw.find("{")
            end = raw.rfind("}")
            payload = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw
            data = json.loads(payload)

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
        fallback = {
            "bias": "NEUTRAL",
            "risk_mode": "NORMAL",
            "entry_style": "MIXED",
            "avoid_new_entries": False,
            "max_hold_minutes": 0,
            "confidence": 0.0,
            "rationale": "AI disabled",
        }
        if not self.enabled:
            return fallback

        import time
        now = time.time()
        if now - getattr(self, "_last_overlay_time", 0.0) < 60:
            if getattr(self, "_last_overlay", None):
                return self._last_overlay
            return fallback
        self._last_overlay_time = now

        system_prompt = (
            "You are a tactical scalp-trading specialist for a crypto bot. "
            "Produce a tactical bias based on 3m, 5m, and 15m charts. "
            "Be decisive: look for BOS/CHoCH (Market Structure) and Pivot bounces. "
            "Favor momentum: if 3m and 5m are trending, set a strong bias (LONG_ONLY or SHORT_ONLY). "
            "Output valid JSON ONLY with keys: bias, risk_mode, entry_style, avoid_new_entries, "
            "max_hold_minutes, confidence, rationale. "
            "bias must be LONG_ONLY, SHORT_ONLY, or NEUTRAL. "
            "risk_mode must be NORMAL, CAUTIOUS, or RISK_OFF. "
            "entry_style must be BUY_PULLBACKS, SELL_RALLIES, BREAKOUTS, or MIXED. "
            "avoid_new_entries must be true or false. max_hold_minutes must be an integer >= 0. "
            "confidence must be 0..1."
        )
        user_content = json.dumps(context, ensure_ascii=False)
        try:
            raw = self._call_llm(system_prompt, user_content, max_tokens=250)
            start = raw.find("{")
            end = raw.rfind("}")
            payload = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw
            data = json.loads(payload)

            bias = str(data.get("bias", "NEUTRAL")).upper()
            if bias not in {"LONG_ONLY", "SHORT_ONLY", "NEUTRAL"}:
                bias = "NEUTRAL"

            risk_mode = str(data.get("risk_mode", "NORMAL")).upper()
            if risk_mode not in {"NORMAL", "CAUTIOUS", "RISK_OFF"}:
                risk_mode = "NORMAL"

            entry_style = str(data.get("entry_style", "MIXED")).upper()
            if entry_style not in {"BUY_PULLBACKS", "SELL_RALLIES", "BREAKOUTS", "MIXED"}:
                entry_style = "MIXED"

            avoid_new_entries = bool(data.get("avoid_new_entries", False))
            max_hold_minutes = int(data.get("max_hold_minutes", 0) or 0)
            if max_hold_minutes < 0:
                max_hold_minutes = 0

            confidence = float(data.get("confidence", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))
            rationale = str(data.get("rationale", "") or "")[:240]

            out = {
                "bias": bias,
                "risk_mode": risk_mode,
                "entry_style": entry_style,
                "avoid_new_entries": avoid_new_entries,
                "max_hold_minutes": max_hold_minutes,
                "confidence": confidence,
                "rationale": rationale,
            }
            self._last_overlay = out
            return out
        except Exception as e:
            logger.error(f"AI overlay evaluation failed: {e}")
            return fallback

    def determine_macro_regime(self, news_headlines: list, quant_context: str = "") -> str:
        if not self.enabled or not news_headlines:
            return "NEUTRAL"

        import time
        now = time.time()
        if now - getattr(self, "_last_regime_time", 0.0) < 60:
            return getattr(self, "last_macro_regime", "NEUTRAL")
        self._last_regime_time = now

        system_prompt = (
            "You are a strict, ultra-conservative risk-management AI for a quantitative crypto trading bot. "
            "Your job is to read the latest news headlines and quantitative metrics, then determine the current market regime. "
            "You MUST reply with exactly ONE word from this list: BULLISH, BEARISH, NEUTRAL, VOLATILE. "
            "Rules: "
            "- If news implies sudden panic, crashes, or unpredictable regulatory action, output VOLATILE. "
            "- If news is overwhelmingly positive, output BULLISH. "
            "- If news is negative or macro environment is tightening, output BEARISH. "
            "- If no clear strong sentiment exists, output NEUTRAL."
        )

        user_content = "Latest Headlines:\n" + "\n".join(news_headlines)
        if quant_context:
            user_content += f"\n\nQuantitative Context:\n{quant_context}"

        try:
            raw = self._call_llm(system_prompt, user_content, max_tokens=10).strip().upper()
            valid_regimes = {"BULLISH", "BEARISH", "VOLATILE", "NEUTRAL"}
            if raw in valid_regimes:
                self.last_macro_regime = raw
            else:
                for reg in valid_regimes:
                    if reg in raw:
                        self.last_macro_regime = reg
                        return self.last_macro_regime
                logger.warning(f"AI returned unexpected regime '{raw}', defaulting to NEUTRAL.")
                self.last_macro_regime = "NEUTRAL"

            return self.last_macro_regime

        except Exception as e:
            logger.error(f"AI regime determination failed: {e}")
            return self.last_macro_regime
