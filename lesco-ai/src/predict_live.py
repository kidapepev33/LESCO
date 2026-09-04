"""Main entry point for continuous LESCO sentence recognition."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from config_ui import open_config_editor
from continuous_recognition import (
    ContinuousRecognizer,
    PrototypeLibrary,
    SentenceBuilder,
    SentenceResult,
    SignDetection,
    StaticSegment,
    sliding_window_ranges,
)
from debug_view import draw_frame, write_debug_response
from feature_extraction import extract_landmark_features, palm_scale, static_landmark_signature
from hand_tracker import HandTracker, select_two_hand_slots
from live_session import (
    ACCELERATION_ACTIVITY_WEIGHT,
    DEFAULT_FPS,
    ONE_HAND_FRAME_SHAPE,
    TWO_HAND_FRAME_SHAPE,
    USE_ACCELERATION_ACTIVITY,
    USE_STATIC_LANDMARK_SIGNATURES,
    CaptureState,
    LandmarkClipRecorder,
    LiveRecognitionSession,
    LiveSessionState,
    RecorderStep,
    ensure_two_hand_frame,
    save_clip_if_needed,
    two_hand_landmarks,
)
from runtime_config import LiveRecognitionConfig, apply_arg_overrides, default_config_path, load_runtime_config
from segment_prediction import (
    RawSegmentDebugEntry,
    SegmentDecision,
    SegmentPrediction,
    SegmentPredictionBuffer,
    classify_segment,
    top_model_predictions,
)
from sign_video_bridge import write_godot_output


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
    print(f"Score lenguaje: {result.language_score:.3f}")
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
    if fps <= 1.0 or fps > 120.0:
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
    project_root = Path(__file__).resolve().parent.parent
    tracker = make_hand_tracker()
    cap = None
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

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = session.process_frame(frame)
            session.publish_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if not session.handle_key(key):
                break
    finally:
        tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Run continuous sentence recognition from camera or an offline clip."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
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
