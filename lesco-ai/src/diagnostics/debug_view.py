"""Text diagnostics for live recognition."""

from __future__ import annotations

from pathlib import Path


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
