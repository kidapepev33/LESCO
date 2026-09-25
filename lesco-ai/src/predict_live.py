"""Compatibility launcher for :mod:`src.recognition.predict_live`."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.recognition.predict_live import *  # noqa: E402,F403
except KeyboardInterrupt:
    if __name__ != "__main__":
        raise
    print("\n[RECOGNITION] Cierre solicitado durante el arranque.")
    raise SystemExit(130) from None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RECOGNITION] Cierre solicitado. Recursos liberados.")
