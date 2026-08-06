"""Camera overlays for normal and debug live recognition."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from continuous_recognition import SentenceResult
from runtime_config import LiveRecognitionConfig


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


def draw_debug_overlay(
    frame: np.ndarray,
    *,
    state: str,
    recorded_frames: int,
    pause_counter: int,
    end_threshold: int,
    movement: float,
    no_hand_frames: int,
    no_hand_threshold: int,
    has_pending_low_confidence: bool,
    detected_segments: int,
    individual_confidence: float | None,
    concatenated_confidence: float | None,
    static_signature: dict[str, object] | None,
    static_score: float | None,
    sentence_status: str,
    config: LiveRecognitionConfig,
    window_count: int = 0,
    processing_ms: float | None = None,
    result: SentenceResult | None = None,
) -> np.ndarray:
    """Draw technical runtime details for debugging."""
    if static_signature is None:
        static_lines = ["Firma estatica: no"]
    else:
        static_status = "aceptada" if bool(static_signature["accepted"]) else "ambigua"
        static_score_text = "-" if static_score is None else f"{static_score:.3f}"
        static_lines = [
            "Firma estatica: si",
            f"Landmark dominante: {int(static_signature['dominant_landmark'])}",
            f"Dominancia: {float(static_signature['dominance']):.1f}% ({static_status})",
            f"Score estatico: {static_score_text}",
        ]
    state_text = f"{state} {pause_counter}/{end_threshold}" if state == "POSSIBLE_PAUSE" else state
    lines = [
        f"Estado: {state_text}",
        f"Pausa: {pause_counter}/{end_threshold}",
        f"Movimiento: {movement:.4f}",
        f"Frames segmento: {recorded_frames}",
        *static_lines,
        f"Sin manos: {no_hand_frames}/{no_hand_threshold}",
    ]
    if result is not None:
        lines.extend(
            [
                f"Oracion: {result.sentence}",
                f"Score visual: {result.visual_score:.3f}",
                f"Score lenguaje: {result.language_score:.3f}",
                f"Candidatos: {len(result.candidates)}",
                f"Detecciones: {len(result.detections)}",
            ]
        )
        for det in result.detections[:4]:
            lines.append(
                f"{det.word.upper()} conf={det.confidence:.2f} "
                f"frames={det.start_frame}-{det.end_frame} support={det.support}"
            )
    return draw_text_lines(frame, lines, color=(0, 255, 0))
