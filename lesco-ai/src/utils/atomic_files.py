"""Cross-platform helpers for safely publishing shared files."""

from __future__ import annotations

import os
import time
from pathlib import Path


_WINDOWS_SHARING_ERRORS = {5, 32, 33}


def _is_temporary_replace_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in _WINDOWS_SHARING_ERRORS


def replace_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 6,
    delay_seconds: float = 0.02,
) -> None:
    """Atomically replace a file, retrying brief Windows sharing violations."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if attempt == attempts - 1 or not _is_temporary_replace_error(error):
                raise
            time.sleep(delay_seconds)


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text through a sibling temporary file, then publish it atomically."""

    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding=encoding)
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
