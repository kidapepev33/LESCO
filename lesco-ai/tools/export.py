#!/usr/bin/env python3
"""Export the complete Prisma project to a portable ZIP archive."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXCLUDED_DIRECTORIES = {".venv", "tests"}


def export_project() -> Path:
    """Create a timestamped project export and return its absolute path."""

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    destination = EXPORTS_DIR / f"Prisma_{timestamp}.zip"

    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for current_directory, directory_names, file_names in os.walk(PROJECT_ROOT):
            current_path = Path(current_directory)
            directory_names[:] = sorted(
                name for name in directory_names if name not in EXCLUDED_DIRECTORIES
            )

            relative_directory = current_path.relative_to(PROJECT_ROOT)
            if relative_directory != Path("."):
                archive.writestr(relative_directory.as_posix().rstrip("/") + "/", b"")

            for file_name in sorted(file_names):
                source = current_path / file_name
                if source.suffix.lower() == ".zip" or source.resolve() == destination.resolve():
                    continue
                archive.write(source, source.relative_to(PROJECT_ROOT).as_posix())

    return destination.resolve()


if __name__ == "__main__":
    print(export_project())
