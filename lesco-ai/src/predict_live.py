"""Compatibility launcher for :mod:`src.recognition.predict_live`."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.process_signals import install_windows_break_handler  # noqa: E402

try:
    from src.recognition.predict_live import *  # noqa: E402,F403
except KeyboardInterrupt:
    if __name__ != "__main__":
        raise
    print("\n[RECOGNITION] Cierre solicitado durante el arranque.")
    raise SystemExit(130) from None


if __name__ == "__main__":
    install_windows_break_handler()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RECOGNITION] Cierre solicitado. Recursos liberados.")
