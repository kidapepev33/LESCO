#!/usr/bin/env python3
"""Genera los iconos PWA cuadrados desde el logo oficial sin deformarlo."""

from pathlib import Path

from PIL import Image


PWA_DIR = Path(__file__).resolve().parent
ASSET_SOURCE = PWA_DIR.parent / "assets" / "images" / "Logopwa.png"
# The self-contained PWA copy prevents broken icon generation in checkouts made
# before Logopwa.png was added to assets/images.
SOURCE = ASSET_SOURCE if ASSET_SOURCE.is_file() else PWA_DIR / "logopwa.png"
OUTPUT_DIR = PWA_DIR / "icons"
BACKGROUND = "#FFFFFF"
CONTENT_RATIO = 0.86


def render(size: int) -> Image.Image:
    source = Image.open(SOURCE).convert("RGBA")
    available = int(size * CONTENT_RATIO)
    scale = min(available / source.width, available / source.height)
    dimensions = (round(source.width * scale), round(source.height * scale))
    source = source.resize(dimensions, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), BACKGROUND)
    position = ((size - source.width) // 2, (size - source.height) // 2)
    canvas.alpha_composite(source, position)
    return canvas


def generate(size: int) -> Path:
    canvas = render(size)
    output = OUTPUT_DIR / f"icon-{size}.png"
    canvas.convert("RGB").save(output, optimize=True)
    return output


def generate_windows_icon() -> Path:
    output = OUTPUT_DIR / "prisma.ico"
    render(256).save(output, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return output


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for icon_size in (192, 512):
        print(generate(icon_size))
    print(generate_windows_icon())
