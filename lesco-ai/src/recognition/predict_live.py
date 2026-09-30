"""Main entry point for continuous LESCO sentence recognition."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src.config.constants import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.config.editor import open_config_editor
from src.recognition.continuous import (
    ContinuousRecognizer,
    PrototypeLibrary,
    SentenceBuilder,
    SentenceResult,
    SignDetection,
    StaticSegment,
    sliding_window_ranges,
)
from src.vision.features import extract_landmark_features, palm_scale, static_landmark_signature
from src.vision.hand_tracker import HandTracker, select_two_hand_slots
from src.recognition.session import LiveRecognitionSession
from src.config.runtime import (
    DEFAULT_FPS,
    MAX_VALID_FPS,
    MIN_VALID_FPS,
    LiveRecognitionConfig,
    apply_arg_overrides,
    default_config_path,
    load_runtime_config,
)
from src.recognition.segments import (
    RawSegmentDebugEntry,
    SegmentDecision,
    SegmentPrediction,
    SegmentPredictionBuffer,
    classify_segment,
    top_model_predictions,
)
from src.integration.sign_video_bridge import write_godot_output


def parse_args() -> argparse.Namespace:
    """Parse command line options for the main recognizer."""
    parser = argparse.ArgumentParser(description="Reconoce oraciones LESCO desde cámara o un clip .npy.")
    parser.add_argument("--input-npy", type=Path, help="Clip de landmarks con shape (frames, 2, 21, 3).")
    parser.add_argument(
        "--record-seconds",
        type=float,
        help="Usa este valor como duración máxima de clip.",
    )
    parser.add_argument("--stride", type=int, help="Override para stride de ventanas temporales.")
    parser.add_argument("--min-confidence", type=float, help="Override para confianza mínima.")
    parser.add_argument("--save-clip", type=Path, help="Guarda clips capturados en .npy.")
    parser.add_argument("--config", action="store_true", help="Abre la configuración local y sale.")
    parser.add_argument("--no-prototypes", action="store_true", help="Desactiva validación gestual por prototipos.")
    return parser.parse_args()


def print_result(result: SentenceResult, processing_ms: float | None = None) -> None:
    """Print the continuous-recognition result."""
    print(f"Oración: {result.sentence}")
    if processing_ms is not None:
        print(f"Procesamiento: {processing_ms:.1f} ms")
    print(f"Score visual: {result.visual_score:.3f}")
    print(f"Ajuste temporal: {result.language_score:.3f}")
    print("Detecciones:")
    for detection in result.detections:
        static = ""
        if detection.static_dominant_landmark is not None:
            status = "aceptada" if detection.static_accepted else "ambigua"
            score = "-" if detection.static_score is None else f"{detection.static_score:.3f}"
            static = (
                f" static_lm={detection.static_dominant_landmark} "
                f"dom={detection.static_dominance:.1f}% {status} static={score}"
            )
        print(
            f"  {detection.word.upper()} conf={detection.confidence:.3f} "
            f"frames={detection.start_frame}-{detection.end_frame} support={detection.support}{static}"
        )
    print("Candidatos:")
    for candidate in result.candidates:
        proto = "" if candidate.prototype_score is None else f", proto={candidate.prototype_score:.3f}"
        static = "" if candidate.static_score is None else f", static={candidate.static_score:.3f}"
        print(
            f"  {candidate.word.upper()} conf={candidate.confidence:.3f} "
            f"frames={candidate.start_frame}-{candidate.end_frame} support={candidate.support}{proto}{static}"
        )


def process_sequence(
    sequence: np.ndarray,
    recognizer: ContinuousRecognizer,
    config: LiveRecognitionConfig,
    static_segments: list[StaticSegment] | None = None,
) -> tuple[SentenceResult, float, int]:
    """Run continuous recognition and return result, time and window count."""
    window_count = len(sliding_window_ranges(len(sequence), stride=config.stride))
    start = time.perf_counter()
    result = recognizer.recognize(
        sequence,
        stride=config.stride,
        min_confidence=config.min_confidence,
        static_segments=static_segments,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms, window_count


def run_offline_clip(args: argparse.Namespace, config: LiveRecognitionConfig, output_text_path: Path) -> None:
    """Recognize a pre-recorded landmark clip."""
    sequence = np.load(args.input_npy)
    prototypes = PrototypeLibrary.from_dataset() if config.use_prototypes else None
    recognizer = ContinuousRecognizer(prototypes=prototypes)
    result, elapsed_ms, _ = process_sequence(sequence, recognizer, config)
    write_godot_output(output_text_path, result)
    print_result(result, processing_ms=elapsed_ms)


def camera_fps(cap: cv2.VideoCapture) -> float:
    """Return camera FPS, falling back to a stable default when invalid."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= MIN_VALID_FPS or fps > MAX_VALID_FPS:
        fps = DEFAULT_FPS
    return fps


def open_live_camera() -> cv2.VideoCapture:
    """Open and configure the live camera."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


class LatestFrameCapture:
    """Captura una sola cámara y conserva únicamente su frame más reciente."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._thread = threading.Thread(target=self._capture_loop, name="prisma-camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                success, frame = self.cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()

    def latest_after(self, sequence: int, timeout: float = 0.5) -> tuple[int, np.ndarray | None]:
        """Devuelve el frame más nuevo, omitiendo intermedios si inferencia va más lenta."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence or not self._thread.is_alive(),
                timeout=timeout,
            )
            if self._sequence <= sequence or self._frame is None:
                return sequence, None
            return self._sequence, self._frame.copy()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self.cap.release()
            self._thread.join(timeout=1.0)


def make_hand_tracker() -> HandTracker:
    """Create the MediaPipe hand tracker used by live recognition."""
    return HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def run_camera(
    args: argparse.Namespace,
    config: LiveRecognitionConfig,
    output_text_path: Path,
    output_frame_path: Path,
    debug_response_path: Path,
) -> None:
    """Run the camera loop until the user quits."""
    project_root = Path(__file__).resolve().parents[2]
    tracker = make_hand_tracker()
    cap = None
    capture = None
    try:
        cap = open_live_camera()
        session = LiveRecognitionSession.create(
            args=args,
            config=config,
            output_text_path=output_text_path,
            output_frame_path=output_frame_path,
            debug_response_path=debug_response_path,
            tracker=tracker,
            fps=camera_fps(cap),
            project_root=project_root,
            result_printer=print_result,
        )
        capture = LatestFrameCapture(cap)
        capture.start()
        sequence = 0

        while True:
            sequence, frame = capture.latest_after(sequence)
            if frame is None:
                if not capture.is_running:
                    break
                continue
            frame = session.process_frame(frame)
            session.show_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if not session.handle_key(key):
                break
    finally:
        if capture is not None:
            capture.stop()
        tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Run continuous sentence recognition from camera or an offline clip."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = default_config_path(project_root)

    if args.config:
        open_config_editor(config_path)
        return

    config = apply_arg_overrides(load_runtime_config(config_path), args)
    godot_bridge_dir = project_root / "godot_bridge"
    godot_bridge_dir.mkdir(exist_ok=True)
    output_text_path = godot_bridge_dir / "output.txt"
    output_frame_path = godot_bridge_dir / "frame.jpg"
    debug_response_path = godot_bridge_dir / "debug_response.txt"

    try:
        if args.input_npy is not None:
            run_offline_clip(args, config, output_text_path)
        else:
            run_camera(args, config, output_text_path, output_frame_path, debug_response_path)
    except Exception as exc:
        output_text_path.write_text(f"Oración: \nError: {exc}", encoding="utf-8")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
