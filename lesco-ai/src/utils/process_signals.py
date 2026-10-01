"""Platform-specific signal setup for Prisma-owned console processes."""

from __future__ import annotations

import os
import signal


def install_windows_break_handler() -> None:
    """Translate a targeted Windows Ctrl+Break into normal Python cleanup."""

    if os.name != "nt" or not hasattr(signal, "SIGBREAK"):
        return

    def request_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGBREAK, request_shutdown)
