from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import numbers
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

EDITABLE_FIELDS = [
    # CATEGORY: EXECUTION ENGINE
    {"key": "execution.mode", "label": "Execution Mode", "type": "select", "options": ["paper", "live"], "cat": "Execution Engine"},
    {"key": "execution.market", "label": "Market Type", "type": "select", "options": ["spot", "usdm"], "cat": "Execution Engine"},
    {"key": "strategy.fixed_trade_usdt", "label": "Trade Size (USDT)", "type": "number", "step": "1", "min": 1, "max": 100000, "cat": "Execution Engine"},
    {"key": "execution.leverage", "label": "Manual Leverage", "type": "number", "step": "1", "min": 1, "max": 100, "cat": "Execution Engine"},
    {"key": "leverage.enabled", "label": "Dynamic Leverage", "type": "bool", "cat": "Execution Engine"},
    {"key": "leverage.max_leverage", "label": "Max Leverage Cap", "type": "number", "step": "1", "min": 1, "max": 100, "cat": "Execution Engine"},
    {"key": "execution.use_limit_orders", "label": "Use Limit Orders", "type": "bool", "cat": "Execution Engine"},
    {"key": "execution.close_positions_on_pause", "label": "Close Positions On Pause", "type": "bool", "cat": "Execution Engine"},
    {"key": "execution.min_seconds_between_trades", "label": "Trade Cooldown (s)", "type": "number", "step": "1", "min": 60, "max": 3600, "cat": "Execution Engine"},

    # CATEGORY: RISK MANAGEMENT
    {"key": "strategy.sl_pct", "label": "Stop Loss %", "type": "number", "step": "0.0001", "min": 0.0001, "max": 1.0, "cat": "Risk Management"},
    {"key": "strategy.tp_pct", "label": "Take Profit %", "type": "number", "step": "0.0001", "min": 0.0001, "max": 1.0, "cat": "Risk Management"},
    {"key": "strategy.max_structural_sl_pct", "label": "Max Structural SL %", "type": "number", "step": "0.0001", "min": 0.0001, "max": 1.0, "cat": "Risk Management"},
    {"key": "execution.break_even_trigger_pct", "label": "B/E Trigger %", "type": "number", "step": "0.0001", "min": 0.0, "max": 1.0, "cat": "Risk Management"},
    {"key": "risk.daily_loss_cap", "label": "Daily Loss Cap", "type": "number", "step": "0.01", "min": 0.0, "max": 1.0, "cat": "Risk Management"},
    {"key": "risk.min_balance_floor", "label": "Balance Floor ($)", "type": "number", "step": "0.01", "min": 0.0, "max": 1e9, "cat": "Risk Management"},

    # CATEGORY: STRATEGY & FILTERS
    {"key": "strategy.min_conf", "label": "Min Confidence", "type": "number", "step": "0.01", "min": 0.15, "max": 1.0, "cat": "Strategy & Filters"},
    {"key": "strategy.max_spread", "label": "Max Spread %", "type": "number", "step": "0.0001", "min": 0.0, "max": 1.0, "cat": "Strategy & Filters"},
    {"key": "strategy.max_chase_pct", "label": "Max Chase %", "type": "number", "step": "0.0001", "min": 0.0, "max": 1.0, "cat": "Strategy & Filters"},
    {"key": "strategy.wick_sweep_enabled", "label": "Wick Sweep Filter", "type": "bool", "cat": "Strategy & Filters"},
    {"key": "mtf.sr_buffer_pct", "label": "S/R Buffer %", "type": "number", "step": "0.0001", "min": 0.0, "max": 1.0, "cat": "Strategy & Filters"},

    # CATEGORY: AI INTELLIGENCE
    {"key": "ai.enabled", "label": "Enable AI Global", "type": "bool", "cat": "AI Intelligence"},
    {"key": "ai.model", "label": "AI Core Model", "type": "select", "options": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"], "cat": "AI Intelligence"},
    {"key": "mtf.min_agree", "label": "MTF Min Agreement", "type": "number", "step": "1", "min": 1, "max": 10, "cat": "AI Intelligence"},
    {"key": "auto_learning.enabled", "label": "Auto-Learning", "type": "bool", "cat": "AI Intelligence"},
    {"key": "auto_learning.ai_advisor.enabled", "label": "AI Auto-Tuning", "type": "bool", "cat": "AI Intelligence"},

    # CATEGORY: SYSTEM CORE
    {"key": "intervals.tick_delay_seconds", "label": "Tick Delay (s)", "type": "number", "step": "0.1", "min": 0.1, "max": 120.0, "cat": "System Core"},
    {"key": "intervals.indicator_refresh", "label": "Indicator Refresh (Ticks)", "type": "number", "step": "1", "min": 1, "max": 1000, "cat": "System Core"},
]

_INDEX_HTML_CACHE: str | None = None
_INDEX_HTML_CACHE_MTIME: float | None = None

def get_index_html() -> str:
    global _INDEX_HTML_CACHE, _INDEX_HTML_CACHE_MTIME
    path = Path(__file__).parent / "static" / "index.html"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[DASHBOARD] Failed to load index.html: {exc}")
    _INDEX_HTML_CACHE = "<h1>Error loading index.html</h1>"
    _INDEX_HTML_CACHE_MTIME = None
    return _INDEX_HTML_CACHE

def _deep_get(data: dict, path: str):
    cur = data
    for part in str(path).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

def _deep_set(data: dict, path: str, value: Any):
    parts = str(path).split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value

def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                return None
            return numeric_value
        except Exception:
            return None
    if isinstance(value, str) and value.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
        return None
    return value

def _coerce_value(field: dict, raw_value: Any, current_value: Any):
    field_type = str(field.get("type", "")).lower()
    if field_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if field_type == "select":
        return str(raw_value)
    if field_type == "number":
        if raw_value in ("", None):
            return current_value
        try:
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                return int(float(raw_value))
            return float(raw_value)
        except (TypeError, ValueError):
            return current_value
    if raw_value in ("", None):
        return current_value
    if isinstance(current_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        try:
            return int(float(raw_value))
        except (TypeError, ValueError):
            return raw_value
    if isinstance(current_value, float):
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return raw_value
    return raw_value

def _validate_value(field: dict, value: Any) -> tuple[bool, str]:
    field_type = str(field.get("type", "")).lower()
    if field_type == "bool":
        return True, ""
    if field_type == "select":
        options = field.get("options", [])
        if value not in options:
            return False, f"Value must be one of: {options}"
        return True, ""
    if field_type == "number":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False, "Value must be a number"
        min_val = field.get("min")
        max_val = field.get("max")
        if min_val is not None and v < min_val:
            return False, f"Value must be >= {min_val}"
        if max_val is not None and v > max_val:
            return False, f"Value must be <= {max_val}"
        return True, ""
    return True, ""

class DashboardRuntime:
    def __init__(self, cfg: dict, config_path: str = "config.yaml", overrides_path: str = "ui_state.json"):
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.overrides_path = Path(overrides_path)
        self.ui_overrides: dict = {}
        self.lock = threading.RLock()
        self.state: dict = {}
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.host: str | None = None
        self.port: int | None = None
        self._last_start_attempt_ts: float = 0.0
        self._auth_token: str | None = os.getenv("DASHBOARD_TOKEN", None)

    def update_state(self, state: dict):
        with self.lock:
            self.state = copy.deepcopy(state or {})

    def get_state(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.state)

    def get_config(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.cfg)

    def save_overrides(self):
        with self.lock:
            try:
                tmp_path = Path(str(self.overrides_path) + ".tmp")
                tmp_path.write_text(json.dumps(self.ui_overrides, indent=2), encoding="utf-8")
                tmp_path.replace(self.overrides_path)
            except Exception as exc:
                print(f"[DASHBOARD] Failed to save overrides: {exc}")

    def apply_settings(self, values: dict) -> dict:
        changed: dict[str, Any] = {}
        valid_keys = {f["key"] for f in EDITABLE_FIELDS}
        field_map = {f["key"]: f for f in EDITABLE_FIELDS}
        with self.lock:
            for key, raw_value in values.items():
                if key not in valid_keys:
                    continue
                field = field_map[key]
                current = _deep_get(self.cfg, key)
                coerced = _coerce_value(field, raw_value, current)
                ok, msg = _validate_value(field, coerced)
                if not ok:
                    continue
                self.ui_overrides[key] = coerced
                changed[key] = coerced
            self.save_overrides()
            for key, coerced in changed.items():
                _deep_set(self.cfg, key, coerced)
        return changed

    def start(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = str(host)
        requested_port = int(port)
        last_error = None
        server = None
        for candidate_port in range(requested_port, requested_port + 10):
            try:
                server = ThreadingHTTPServer((self.host, candidate_port), _DashboardHandler)
                self.port = int(candidate_port)
                break
            except OSError as exc:
                last_error = exc
                continue
        if server is None:
            raise last_error or OSError(f"Unable to bind dashboard on {self.host}:{requested_port}")
        server.runtime = self
        self.httpd = server
        self.thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        self.thread.start()
        self._last_start_attempt_ts = time.time()
        print(f"[DASHBOARD] Web dashboard running on http://{self.host}:{self.port}")
        return server

    def ensure_running(self, host: str | None = None, port: int | None = None) -> bool:
        host = str(host or self.host or "0.0.0.0")
        port = int(port or self.port or 8765)
        if self.thread is not None and self.thread.is_alive() and self.httpd is not None:
            return True
        now = time.time()
        if (now - self._last_start_attempt_ts) < 10.0:
            return False
        try:
            self.start(host=host, port=port)
            return True
        except Exception as exc:
            print(f"[DASHBOARD] Failed to start: {exc}")
            self._last_start_attempt_ts = now
            return False

class _DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        status = None
        if len(args) >= 2:
            try:
                status = int(args[1])
            except (TypeError, ValueError):
                status = None
        if status is not None and status < 400:
            return
        print(f"[DASHBOARD] {self.client_address[0]} - {format % args}")

    @property
    def runtime(self) -> DashboardRuntime:
        return getattr(self.server, "runtime")

    def _check_auth(self) -> bool:
        token = self.runtime._auth_token
        if not token:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return hmac.compare_digest(auth_header[7:], token)
        return False

    def _send(self, payload: Any, status: int = 200, content_type: str = "application/json"):
        body = payload if isinstance(payload, (bytes, bytearray)) else str(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def do_GET(self):
        if not self._check_auth():
            self._send(json.dumps({"error": "unauthorized"}).encode("utf-8"), status=401)
            return
        if self.path in {"/", "/index.html"}:
            self._send(get_index_html(), content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send(json.dumps(_json_safe(self.runtime.get_state()), default=str, allow_nan=False).encode("utf-8"))
            return
        if self.path == "/api/config":
            self._send(json.dumps(_json_safe(self.runtime.get_config()), default=str, allow_nan=False).encode("utf-8"))
            return
        if self.path == "/api/schema":
            self._send(json.dumps(_json_safe(EDITABLE_FIELDS), allow_nan=False).encode("utf-8"))
            return
        self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def do_POST(self):
        if not self._check_auth():
            self._send(json.dumps({"error": "unauthorized"}).encode("utf-8"), status=401)
            return
        if self.path == "/api/toggle-pause":
            try:
                with self.runtime.lock:
                    is_paused = not bool(self.runtime.cfg.get("execution", {}).get("paused", False))
                    if "execution" not in self.runtime.cfg:
                        self.runtime.cfg["execution"] = {}
                    self.runtime.cfg["execution"]["paused"] = is_paused
                    if "config" in self.runtime.state:
                        if "execution" not in self.runtime.state["config"]:
                            self.runtime.state["config"]["execution"] = {}
                        self.runtime.state["config"]["execution"]["paused"] = is_paused
                    print(f"[DASHBOARD] Toggle-Pause: new state = {is_paused}")
                self._send(json.dumps({"ok": True, "paused": is_paused}).encode("utf-8"))
            except Exception as exc:
                self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)
            return
        if self.path == "/api/command":
            try:
                data = self._read_json_body()
                cmd = data.get("cmd")
                if cmd == "set_tf":
                    tf = data.get("tf", "5m")
                    with self.runtime.lock:
                        if "ui" not in self.runtime.cfg:
                            self.runtime.cfg["ui"] = {}
                        self.runtime.cfg["ui"]["chart_tf"] = tf
                    print(f"[DASHBOARD] Set-TF: {tf}")
                    self._send(json.dumps({"ok": True}).encode("utf-8"))
                else:
                    self._send(json.dumps({"ok": False, "error": "unknown command"}).encode("utf-8"), status=400)
            except Exception as exc:
                self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)
            return
        if self.path == "/api/emergency-exit":
            try:
                with self.runtime.lock:
                    if "execution" not in self.runtime.cfg:
                        self.runtime.cfg["execution"] = {}
                    self.runtime.cfg["execution"]["panic_exit"] = True
                print("[DASHBOARD] EMERGENCY EXIT TRIGGERED!")
                self._send(json.dumps({"ok": True, "message": "Emergency signal sent"}).encode("utf-8"))
            except Exception as exc:
                self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)
            return
        if self.path == "/api/settings":
            try:
                data = self._read_json_body()
                values = data.get("values", {})
                changed = self.runtime.apply_settings(values)
                with self.runtime.lock:
                    if "config" in self.runtime.state:
                        for k in changed:
                            _deep_set(self.runtime.state["config"], k, changed[k])
                self._send(json.dumps({"ok": True, "changed": dict(changed)}).encode("utf-8"))
            except Exception as exc:
                self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)
            return
        self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)

def start_dashboard_server(runtime: DashboardRuntime, host: str = "0.0.0.0", port: int = 8765):
    return runtime.start(host=host, port=port)
