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
    sentence_status: str,
    config: LiveRecognitionConfig,
    window_count: int = 0,
    processing_ms: float | None = None,
    result: SentenceResult | None = None,
) -> np.ndarray:
    """Draw technical runtime details for debugging."""
    individual_text = "-" if individual_confidence is None else f"{individual_confidence:.3f}"
    concatenated_text = "-" if concatenated_confidence is None else f"{concatenated_confidence:.3f}"
    state_text = f"{state} {pause_counter}/{end_threshold}" if state == "POSSIBLE_PAUSE" else state
    lines = [
        f"Estado: {state_text}",
        f"Pausa: {pause_counter}/{end_threshold}",
        f"Movimiento: {movement:.4f}",
        f"Frames segmento: {recorded_frames}",
        f"Sin manos: {no_hand_frames}/{no_hand_threshold}",
        f"Ventanas detectadas: {detected_segments}",
        f"Pendiente baja confianza: {'si' if has_pending_low_confidence else 'no'}",
        f"Conf individual: {individual_text}",
        f"Conf concatenada: {concatenated_text}",
        f"Cierre: {sentence_status or '-'}",
        f"Ventanas: {window_count} | stride={config.stride}",
        f"Conf min: {config.min_confidence:.2f}",
        f"Prototipos: {'si' if config.use_prototypes else 'no'}",
    ]
    if processing_ms is not None:
        lines.append(f"Procesamiento: {processing_ms:.1f} ms")
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
