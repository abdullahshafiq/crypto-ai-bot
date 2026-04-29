import os
import logging
import json
from openai import OpenAI

logger = logging.getLogger(__name__)


class HybridAIOrchestrator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.last_macro_regime = "NEUTRAL"
        self._last_trade_eval = None
        self._last_overlay = None

        api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(api_key and api_key != "your_openai_api_key_here")

        if self.enabled:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"AI Orchestrator initialized with model: {model}")
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. AI override disabled (defaulting to NEUTRAL).")

    def evaluate_trade(self, context: dict, model: str | None = None) -> dict:
        """
        Evaluates a proposed trade with full context (signal + MTF + S/R + fees).
        Returns:
          {
            "decision": "ALLOW" | "VETO",
            "hold_minutes": int,
            "confidence": float (0..1),
            "rationale": str
          }
        """
        fallback = {"decision": "ALLOW", "hold_minutes": 0, "confidence": 0.0, "rationale": "AI disabled"}
        if not self.enabled or self.client is None:
            return fallback

        chosen_model = model or self.model

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
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
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
            if confidence < 0.0:
                confidence = 0.0
            if confidence > 1.0:
                confidence = 1.0

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
        Produces a slower tactical overlay for the bot.
        Returns:
          {
            "bias": "LONG_ONLY" | "SHORT_ONLY" | "NEUTRAL",
            "risk_mode": "NORMAL" | "CAUTIOUS" | "RISK_OFF",
            "entry_style": "BUY_PULLBACKS" | "SELL_RALLIES" | "BREAKOUTS" | "MIXED",
            "avoid_new_entries": bool,
            "max_hold_minutes": int,
            "confidence": float,
            "rationale": str
          }
        """
        fallback = {
            "bias": "NEUTRAL",
            "risk_mode": "NORMAL",
            "entry_style": "MIXED",
            "avoid_new_entries": False,
            "max_hold_minutes": 0,
            "confidence": 0.0,
            "rationale": "AI disabled",
        }
        if not self.enabled or self.client is None:
            return fallback

        chosen_model = model or self.model
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
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=250,
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
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
            return {
                "bias": "NEUTRAL",
                "risk_mode": "NORMAL",
                "entry_style": "MIXED",
                "avoid_new_entries": False,
                "max_hold_minutes": 0,
                "confidence": 0.0,
                "rationale": f"AI error: {e}",
            }

    def determine_macro_regime(self, news_headlines: list, quant_context: str = "") -> str:
        """
        Uses LLM to assess if macro conditions are safe for low-risk quant trading.
        Returns exactly one of: BULLISH, BEARISH, NEUTRAL, VOLATILE.
        """
        if not self.enabled:
            return "NEUTRAL"
            
        if not news_headlines:
            return "NEUTRAL"

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=10,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip().upper()

            # Validate the response is one of the expected regimes
            valid_regimes = {"BULLISH", "BEARISH", "VOLATILE", "NEUTRAL"}
            if raw in valid_regimes:
                self.last_macro_regime = raw
            else:
                logger.warning(f"AI returned unexpected regime '{raw}', defaulting to NEUTRAL.")
                self.last_macro_regime = "NEUTRAL"

            return self.last_macro_regime

        except Exception as e:
            logger.error(f"AI regime determination failed: {e}")
            return self.last_macro_regime  # Return last known regime on error
