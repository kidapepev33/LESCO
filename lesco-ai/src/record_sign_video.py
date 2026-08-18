"""Record raw sign videos for the Godot video database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import time

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH

DEFAULT_FPS = 30.0
WINDOW_NAME = "LESCO-AI | Grabador de videos"


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Graba un video MP4 de una seña LESCO.")
    parser.add_argument("--label", required=True, help="Nombre de la seña, por ejemplo: agua")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    """Return a safe lowercase filename stem for one sign label."""
    safe = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    safe = safe.strip("_")
    if not safe:
        raise ValueError("El label no puede quedar vacío.")
    return safe


def project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parent.parent


def camera_fps(cap) -> float:
    """Return camera FPS, falling back to a stable default when invalid."""
    import cv2

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or fps > 120.0:
        return DEFAULT_FPS
    return fps


def make_writer(path: Path, fps: float):
    """Create an MP4 writer using a broadly supported codec."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))


def draw_overlay(frame, label: str, is_recording: bool, recorded_seconds: float):
    """Draw recorder state over a display frame."""
    import cv2

    state = "GRABANDO" if is_recording else "ESPERANDO"
    color = (0, 0, 255) if is_recording else (0, 220, 255)
    lines = [
        f"Sena: {label.upper()}",
        f"Estado: {state}",
        f"Tiempo: {recorded_seconds:.1f}s",
        "R: comenzar | S: guardar | Q: salir",
    ]
    y = 34
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30
    return frame


def record_video(label: str) -> Path | None:
    """Run the interactive camera recorder and return the saved path."""
    import cv2

    safe_label = sanitize_label(label)
    output_dir = project_root() / "videos_database"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{safe_label}.mp4"
    temp_path = output_dir / f".{safe_label}.tmp.mp4"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    writer: cv2.VideoWriter | None = None
    saved_path: Path | None = None
    recorded_frames = 0
    recording_started_at: float | None = None

    try:
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        fps = camera_fps(cap)

        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("No se pudo leer un frame de la cámara.")

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            is_recording = writer is not None
            recorded_seconds = 0.0
            if is_recording and recording_started_at is not None:
                recorded_seconds = time.perf_counter() - recording_started_at
                writer.write(frame)
                recorded_frames += 1

            display = draw_overlay(frame.copy(), safe_label, is_recording, recorded_seconds)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")) and writer is None:
                temp_path.unlink(missing_ok=True)
                writer = make_writer(temp_path, fps)
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError("No se pudo crear el archivo de video.")
                recorded_frames = 0
                recording_started_at = time.perf_counter()

            if key in (ord("s"), ord("S")) and writer is not None:
                writer.release()
                writer = None
                if recorded_frames > 0:
                    os.replace(temp_path, output_path)
                    saved_path = output_path
                else:
                    temp_path.unlink(missing_ok=True)
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        if saved_path is None:
            temp_path.unlink(missing_ok=True)

    return saved_path


def main() -> None:
    """Run the video recorder."""
    args = parse_args()
    saved_path = record_video(args.label)
    if saved_path is None:
        print("Grabación cancelada. No se guardó ningún video.")
    else:
        print(f"Video guardado: {saved_path}")


if __name__ == "__main__":
    main()
