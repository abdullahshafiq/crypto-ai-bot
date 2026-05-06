import os
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    try:
        import json
        overrides_path = "ui_state.json"
        if path != "config.yaml":
            base = os.path.splitext(os.path.basename(path))[0]
            overrides_path = f"{base}_state.json"
        if os.path.exists(overrides_path):
            with open(overrides_path, 'r') as f:
                ui_state = json.load(f)

            for path_key, val in ui_state.items():
                parts = path_key.split(".")
                cur = cfg
                for part in parts[:-1]:
                    if part not in cur:
                        cur[part] = {}
                    cur = cur[part]
                cur[parts[-1]] = val
    except Exception as e:
        logging.getLogger("main").warning(f"Failed to load overrides from {overrides_path}: {e}")

    return cfg

def _instance_port_for_config(cfg: dict) -> int:
    env_port = os.getenv("BOT_INSTANCE_PORT")
    if env_port:
        return int(env_port)
    exec_mode = str((cfg.get("execution", {}) or {}).get("mode", "live") or "live").strip().lower()
    return 45679 if exec_mode == "paper" else 45678
