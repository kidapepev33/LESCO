"""Camera overlay helpers for live recognition."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

def draw_text_lines(
    frame: np.ndarray,
    lines: Iterable[str],
    origin: tuple[int, int] = (20, 34),
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw compact OpenCV text lines."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 26
    return frame


def draw_normal_overlay(frame: np.ndarray, state: str, sentence: str = "") -> np.ndarray:
    """Draw only presentation-friendly information."""
    lines = [state]
    if sentence:
        lines.append(sentence)
    return draw_text_lines(frame, lines, color=(0, 255, 255))
