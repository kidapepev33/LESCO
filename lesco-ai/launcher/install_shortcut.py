#!/usr/bin/env python3
"""Instala un acceso directo de escritorio para el checkout actual."""

from pathlib import Path
import os


LAUNCHER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAUNCHER_DIR.parent
APPLICATIONS_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "applications"
DESKTOP_PATH = APPLICATIONS_DIR / "prisma.desktop"


def desktop_quote(value: Path) -> str:
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def install() -> Path:
    runner = LAUNCHER_DIR / "run_prisma.sh"
    icon = PROJECT_ROOT / "web" / "pwa" / "icons" / "icon-512.png"
    if not runner.is_file() or not icon.is_file():
        raise FileNotFoundError("Faltan run_prisma.sh o el icono PWA de Prisma.")

    APPLICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    contents = "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=Prisma",
        "Comment=Inicia y abre el sistema local Prisma",
        f"Exec={desktop_quote(runner)}",
        f"Icon={icon}",
        "Terminal=false",
        "Categories=Education;Accessibility;",
        "StartupNotify=true",
        "",
    ])
    DESKTOP_PATH.write_text(contents, encoding="utf-8")
    DESKTOP_PATH.chmod(0o755)
    return DESKTOP_PATH


if __name__ == "__main__":
    installed = install()
    print(f"Acceso directo instalado en: {installed}")
