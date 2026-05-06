import os
import sys
import socket


_SINGLETON_SOCKET = None
def _enforce_single_instance(port=45678):
    global _SINGLETON_SOCKET
    try:
        _SINGLETON_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _SINGLETON_SOCKET.bind(("127.0.0.1", port))
    except OSError:
        print(f"ERROR: Another instance of the bot is already running. Please close it first.")
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
