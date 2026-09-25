"""Compatibility launcher for :mod:`src.integration.sign_video_bridge`."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integration.sign_video_bridge import *  # noqa: E402,F403


if __name__ == "__main__":
    main()
