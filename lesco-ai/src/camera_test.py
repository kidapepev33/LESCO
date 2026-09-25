"""Compatibility launcher for :mod:`src.diagnostics.camera_test`."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnostics.camera_test import main  # noqa: E402


if __name__ == "__main__":
    main()
