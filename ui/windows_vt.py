import os
import sys


_WIN_VT_ENABLED: bool | None = None
_ALT_SCREEN_ENTERED: bool = False
_CURSOR_HIDDEN: bool = False
_LAST_FRAME_LINES: int = 0

# --- Global Color Palette ---
use_ansi = True # Enabled for modern Windows/Linux
CYAN    = '\033[96m'
MAGENTA = '\033[95m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
GREEN   = '\033[92m'
RED     = '\033[91m'
BOLD    = '\033[1m'
RESET   = '\033[0m'
CL      = '\033[K' # Clear line

def _get_windows_console_handle():
    if os.name != "nt":
        return None
    try:
        import ctypes  # type: ignore
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return None
        return handle
    except Exception:
        return None


def _windows_cursor_home() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes  # type: ignore

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short),
                ("Top", ctypes.c_short),
                ("Right", ctypes.c_short),
                ("Bottom", ctypes.c_short),
            ]

        class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
            _fields_ = [
                ("dwSize", COORD),
                ("dwCursorPosition", COORD),
                ("wAttributes", ctypes.c_ushort),
                ("srWindow", SMALL_RECT),
                ("dwMaximumWindowSize", COORD),
            ]

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = _get_windows_console_handle()
        if handle is None:
            return False
        info = CONSOLE_SCREEN_BUFFER_INFO()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        home = COORD(0, info.srWindow.Top)
        return bool(kernel32.SetConsoleCursorPosition(handle, home))
    except Exception:
        return False


def _enable_windows_vt_mode() -> bool:
    """
    Best-effort enabling of ANSI escape sequences on Windows consoles.
    Returns True if VT processing was enabled, else False.
    """
    global _WIN_VT_ENABLED
    if _WIN_VT_ENABLED is not None:
        return _WIN_VT_ENABLED

    if os.name != "nt":
        _WIN_VT_ENABLED = True
        return True

    try:
        import ctypes  # type: ignore

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False

        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False

        _WIN_VT_ENABLED = True
        return True
    except Exception:
        _WIN_VT_ENABLED = False
        return False


def _enter_alt_screen_ansi():
    global _ALT_SCREEN_ENTERED, _CURSOR_HIDDEN
    if _ALT_SCREEN_ENTERED:
        return
    # Alternate screen buffer + hide cursor (prevents scrollback spam).
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()
    _ALT_SCREEN_ENTERED = True
    _CURSOR_HIDDEN = True


def _exit_alt_screen_ansi():
    global _ALT_SCREEN_ENTERED, _CURSOR_HIDDEN
    if not _ALT_SCREEN_ENTERED:
        return
    # Show cursor + leave alternate buffer.
    sys.stdout.write("\033[?25h\033[?1049l")
    sys.stdout.flush()
    _ALT_SCREEN_ENTERED = False
    _CURSOR_HIDDEN = False


def _hide_cursor_ansi():
    global _CURSOR_HIDDEN
    if _CURSOR_HIDDEN:
        return
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    _CURSOR_HIDDEN = True


def _show_cursor_ansi():
    global _CURSOR_HIDDEN
    if not _CURSOR_HIDDEN:
        return
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
    _CURSOR_HIDDEN = False


def _detect_ui_mode(cfg_mode: str) -> str:
    # 1. Non-interactive terminals (piping, IDE outputs) MUST fallback safely
    if not sys.stdout.isatty():
        return "cls" if os.name == "nt" else "plain"

    # 2. FORCE 'ansi' mode on all interactive terminals that support it.
    # This prevents the UI from breaking or scrolling even if the user
    # misconfigures 'ui.mode' in their config.yaml.
    if os.name == "nt":
        if _enable_windows_vt_mode():
            return "ansi"
        if _get_windows_console_handle() is not None:
            return "win"
        return "cls"

    return "ansi"
