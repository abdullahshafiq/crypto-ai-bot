import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core.bot_loop import run_hybrid_bot

if __name__ == "__main__":
    run_hybrid_bot()
