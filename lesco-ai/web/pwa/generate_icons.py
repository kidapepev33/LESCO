#!/usr/bin/env python3
"""Genera los iconos PWA cuadrados desde el logo oficial sin deformarlo."""

from pathlib import Path

from PIL import Image


PWA_DIR = Path(__file__).resolve().parent
SOURCE = PWA_DIR.parent / "assets" / "images" / "Logopwa.png"
OUTPUT_DIR = PWA_DIR / "icons"
BACKGROUND = "#FFFFFF"
CONTENT_RATIO = 0.86


def generate(size: int) -> Path:
    source = Image.open(SOURCE).convert("RGBA")
    available = int(size * CONTENT_RATIO)
    scale = min(available / source.width, available / source.height)
    dimensions = (round(source.width * scale), round(source.height * scale))
    source = source.resize(dimensions, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), BACKGROUND)
    position = ((size - source.width) // 2, (size - source.height) // 2)
    canvas.alpha_composite(source, position)
    output = OUTPUT_DIR / f"icon-{size}.png"
    canvas.convert("RGB").save(output, optimize=True)
    return output


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for icon_size in (192, 512):
        print(generate(icon_size))
