"""Compatibility launcher for :mod:`src.integration.sign_video_bridge`."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.process_signals import install_windows_break_handler  # noqa: E402

from src.integration.sign_video_bridge import *  # noqa: E402,F403


if __name__ == "__main__":
    install_windows_break_handler()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SIGN VIDEO] Cierre solicitado.")
