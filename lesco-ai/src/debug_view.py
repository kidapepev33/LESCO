"""Camera overlay helpers for live recognition."""

from __future__ import annotations

from pathlib import Path
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


def draw_frame(
    frame: np.ndarray,
    recorder: object,
    config: object,
    result: object | None,
    window_count: int,
    processing_ms: float | None,
    segment_buffer: object | None = None,
    sentence_status: str = "",
) -> np.ndarray:
    """Draw normal camera information."""
    sentence = "" if result is None else result.sentence
    return draw_normal_overlay(frame, recorder.state.value, sentence=sentence)


def write_debug_response(
    debug_path: Path,
    segment_buffer: object,
    final_result: object | None = None,
) -> None:
    """Write raw live segment predictions without changing recognition behavior."""
    lines = ["=== PREDICCIONES CRUDAS ==="]
    for entry in segment_buffer.raw_debug_entries:
        frame_range = ""
        if entry.start_frame is not None and entry.end_frame is not None:
            frame_range = f" | RANGO: {entry.start_frame}-{entry.end_frame}"
        lines.append(f"SEGMENTO {entry.segment_number} | FRAMES: {entry.frame_count}{frame_range}")
        for rank, (word, probability) in enumerate(entry.top_predictions, start=1):
            lines.append(f"{rank}. {word.upper()}: {probability:.2f}")
        if entry.accepted:
            lines.append("DECISION: ACEPTADO")
        else:
            lines.append(f"DECISION: RECHAZADO | UMBRAL: {entry.threshold:.2f}")
        lines.append("")

    lines.append("=== PALABRAS ACEPTADAS ===")
    accepted_words = [word.upper() for segment in segment_buffer.accepted_segments for word in segment.result.words]
    lines.append(" ".join(accepted_words))
    lines.append("")

    lines.append("=== SALIDA FINAL ===")
    lines.append("" if final_result is None else final_result.sentence)

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text("\n".join(lines), encoding="utf-8")
