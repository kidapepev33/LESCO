#!/usr/bin/env python3
"""Create Prisma shortcuts in the Windows Start menu and Desktop."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


LAUNCHER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAUNCHER_DIR.parent


def shortcut_locations() -> tuple[Path, Path]:
    app_data = Path(os.environ["APPDATA"])
    user_profile = Path(os.environ["USERPROFILE"])
    start_menu = app_data / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Prisma.lnk"
    desktop = user_profile / "Desktop" / "Prisma.lnk"
    return start_menu, desktop


def install() -> tuple[Path, Path]:
    if os.name != "nt":
        raise RuntimeError("Este instalador de accesos directos debe ejecutarse en Windows.")

    runner = LAUNCHER_DIR / "run_prisma.cmd"
    icon = PROJECT_ROOT / "web" / "pwa" / "icons" / "prisma.ico"
    if not runner.is_file() or not icon.is_file():
        raise FileNotFoundError("Faltan run_prisma.cmd o web/pwa/icons/prisma.ico.")

    script = (
        "& { param($destination,$runner,$root,$icon);"
        "$shortcut=(New-Object -ComObject WScript.Shell).CreateShortcut($destination);"
        "$shortcut.TargetPath=$runner;"
        "$shortcut.WorkingDirectory=$root;"
        "$shortcut.IconLocation=$icon;"
        "$shortcut.Description='Inicia y abre el sistema local Prisma';"
        "$shortcut.Save() }"
    )
    locations = shortcut_locations()
    for destination in locations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
                str(destination),
                str(runner),
                str(PROJECT_ROOT),
                str(icon),
            ],
            check=True,
        )
    return locations


if __name__ == "__main__":
    for installed in install():
        print(f"Acceso directo instalado en: {installed}")
