"""Bridge sign names from Godot to recorded sign video frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import time

DEFAULT_FPS = 30.0
POLL_INTERVAL_SECONDS = 0.05


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Reproduce videos de señas hacia Godot como JPG.")
    return parser.parse_args()


def project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parent.parent


def sanitize_label(label: str) -> str:
    """Return a safe lowercase filename stem for one sign label."""
    safe = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return safe.strip("_")


def read_requested_label(input_path: Path) -> str:
    """Read and normalize the current sign requested by Godot."""
    if not input_path.exists():
        return ""
    return sanitize_label(input_path.read_text(encoding="utf-8").strip())


def video_fps(capture) -> float:
    """Return video FPS, falling back to a stable default when invalid."""
    import cv2

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or fps > 120.0:
        return DEFAULT_FPS
    return fps


def write_frame_atomic(output_path: Path, temp_path: Path, frame) -> None:
    """Encode one frame as JPG and publish it atomically."""
    import cv2

    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("No se pudo codificar el frame como JPG.")
    temp_path.write_bytes(encoded.tobytes())
    os.replace(temp_path, output_path)


def open_video(video_path: Path):
    """Open a video file and return its capture, or None when unavailable."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def clear_requested_label(input_path: Path) -> None:
    """Clear the current sign request after playback finishes."""
    input_path.write_text("", encoding="utf-8")


def run_bridge() -> None:
    """Continuously mirror requested sign videos into Godot frame output."""
    import cv2

    root = project_root()
    bridge_dir = root / "godot_bridge"
    videos_dir = root / "videos_database"
    input_path = bridge_dir / "sign_video_input.txt"
    output_path = bridge_dir / "sign_video_frame.jpg"
    temp_path = bridge_dir / ".sign_video_frame.tmp.jpg"
    bridge_dir.mkdir(exist_ok=True)

    current_label = ""
    current_video_path: Path | None = None
    capture = None
    frame_delay = 1.0 / DEFAULT_FPS
    next_frame_at = time.perf_counter()

    try:
        while True:
            requested_label = read_requested_label(input_path)
            if requested_label != current_label:
                if capture is not None:
                    capture.release()
                    capture = None
                current_label = requested_label
                current_video_path = None

                if current_label:
                    current_video_path = videos_dir / f"{current_label}.mp4"
                    if not current_video_path.exists():
                        print(f"[SIGN VIDEO] No existe video para '{current_label}': {current_video_path}")
                    else:
                        capture = open_video(current_video_path)
                        if capture is None:
                            print(f"[SIGN VIDEO] No se pudo abrir el video: {current_video_path}")
                        else:
                            frame_delay = 1.0 / video_fps(capture)
                            next_frame_at = time.perf_counter()
                            print(f"[SIGN VIDEO] Reproduciendo: {current_video_path}")

            if not current_label or capture is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            now = time.perf_counter()
            if now < next_frame_at:
                time.sleep(min(POLL_INTERVAL_SECONDS, next_frame_at - now))
                continue

            success, frame = capture.read()
            if not success:
                capture.release()
                capture = None
                print(f"[SIGN VIDEO] Video finalizado: {current_video_path}")
                current_label = ""
                current_video_path = None
                clear_requested_label(input_path)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            write_frame_atomic(output_path, temp_path, frame)
            next_frame_at = time.perf_counter() + frame_delay
    except KeyboardInterrupt:
        print("\n[SIGN VIDEO] Cerrando puente de videos.")
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        temp_path.unlink(missing_ok=True)


def main() -> None:
    """Run the sign video bridge."""
    parse_args()
    run_bridge()


if __name__ == "__main__":
    main()
