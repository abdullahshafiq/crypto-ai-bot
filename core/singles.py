import atexit
import os
import sys
from pathlib import Path


def _lock_path_for_port(port: int) -> Path:
    return Path(f"bot_{port}.lock")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release_instance_lock(lock_path: Path):
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError:
        pass


def _enforce_single_instance(port=45678):
    lock_path = _lock_path_for_port(int(port))

    current_pid = os.getpid()
    if lock_path.exists():
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            locked_pid = int(raw)
        except Exception:
            locked_pid = None

        if locked_pid and locked_pid != current_pid and _pid_is_running(locked_pid):
            print(
                "ERROR: Another instance of the bot is already running "
                f"for port {int(port)} (pid {locked_pid}). Please close it first."
            )
            sys.exit(1)

        _release_instance_lock(lock_path)

    try:
        lock_path.write_text(str(current_pid), encoding="utf-8")
        atexit.register(_release_instance_lock, lock_path)
    except OSError:
        print("ERROR: Unable to create instance lock file. Please check file permissions.")
        sys.exit(1)


def _count_consecutive_losses(closed_trades) -> int:
    losses = 0
    for trade in reversed(list(closed_trades or [])):
        try:
            pnl = float(trade.get("pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl < 0:
            losses += 1
        else:
            break
    return losses
