"""Main entry point for continuous LESCO sentence recognition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
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
from continuous_recognition import ContinuousRecognizer, PrototypeLibrary, SentenceResult, SignDetection, sliding_window_ranges
from debug_view import draw_debug_overlay, draw_normal_overlay
from feature_extraction import extract_landmark_features, palm_scale
from hand_tracker import HandTracker
from runtime_config import LiveRecognitionConfig, default_config_path, load_runtime_config

DEFAULT_FPS = 30.0


class CaptureState(str, Enum):
    """Camera capture state."""

    WAITING = "WAITING"
    MOVING = "MOVING"
    POSSIBLE_PAUSE = "POSSIBLE_PAUSE"


@dataclass
class RecorderStep:
    """Result produced by one state-machine update."""

    state: CaptureState
    finalized_clip: np.ndarray | None = None
    finalized_sentence_clip: np.ndarray | None = None
    discarded_too_short: bool = False
    reached_max_duration: bool = False
    sentence_ended: bool = False


@dataclass
class LandmarkClipRecorder:
    """State machine that segments a sentence clip from hand presence."""

    config: LiveRecognitionConfig
    fps: float = DEFAULT_FPS
    state: CaptureState = CaptureState.WAITING
    no_hand_frames: int = 0
    pause_counter: int = 0
    observed_recording_frames: int = 0
    current_movement: float = 0.0
    clip_frames: list[np.ndarray] = field(default_factory=list)
    sentence_frames: list[np.ndarray] = field(default_factory=list)
    previous_landmarks: np.ndarray | None = None
    sentence_end_reported: bool = False

    @property
    def end_threshold(self) -> int:
        return self.config.pause_frames

    @property
    def no_hand_threshold(self) -> int:
        return max(1, int(round(self.config.no_hands_timeout_seconds * self.fps)))

    @property
    def min_clip_frames(self) -> int:
        return max(1, int(round(self.config.min_clip_seconds * self.fps)))

    @property
    def max_clip_frames(self) -> int:
        return max(1, int(round(self.config.max_clip_seconds * self.fps)))

    @property
    def recorded_frames(self) -> int:
        return len(self.clip_frames)

    @property
    def sentence_recorded_frames(self) -> int:
        return len(self.sentence_frames)

    def reset(self) -> None:
        """Return to WAITING and clear transient buffers."""
        self.state = CaptureState.WAITING
        self.no_hand_frames = 0
        self.pause_counter = 0
        self.observed_recording_frames = 0
        self.current_movement = 0.0
        self.clip_frames.clear()
        self.sentence_frames.clear()
        self.previous_landmarks = None

    def step(self, landmarks: np.ndarray | None) -> RecorderStep:
        """Advance the recorder using one camera frame."""
        has_hand = landmarks is not None
        frame = np.asarray(landmarks, dtype=np.float32) if has_hand else None
        movement = self._movement(frame)
        self.current_movement = movement
        is_active = movement >= self.config.movement_threshold

        if not has_hand:
            self.no_hand_frames += 1
            self.current_movement = 0.0
            if self.no_hand_frames >= self.no_hand_threshold and not self.sentence_end_reported:
                return self._finish_sentence()
            return RecorderStep(self.state)

        self.no_hand_frames = 0
        self.sentence_end_reported = False
        self.sentence_frames.append(frame)

        if self.state == CaptureState.WAITING:
            if not is_active:
                return RecorderStep(self.state)

            self.state = CaptureState.MOVING
            self.pause_counter = 0
            self.observed_recording_frames = 1
            self.clip_frames = [frame]
            return RecorderStep(self.state)

        if self.state in {CaptureState.MOVING, CaptureState.POSSIBLE_PAUSE}:
            self.observed_recording_frames += 1
            self.clip_frames.append(frame)

            if is_active:
                self.state = CaptureState.MOVING
                self.pause_counter = 0
            else:
                self.state = CaptureState.POSSIBLE_PAUSE
                self.pause_counter += 1

            if self.observed_recording_frames >= self.max_clip_frames:
                return self._finalize(reached_max_duration=True)
            if self.pause_counter >= self.end_threshold:
                return self._finalize()
            return RecorderStep(self.state)

        return RecorderStep(self.state)

    def _movement(self, landmarks: np.ndarray | None) -> float:
        if landmarks is None:
            return 0.0

        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 0.0

        scale = max((palm_scale(self.previous_landmarks) + palm_scale(landmarks)) / 2.0, 1e-6)
        velocity = ((landmarks - self.previous_landmarks) / scale).astype(np.float32)
        self.previous_landmarks = landmarks
        return self._mean_landmark_norm(velocity)

    @staticmethod
    def _mean_landmark_norm(values: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(values, axis=1)))

    def _finalize(self, reached_max_duration: bool = False) -> RecorderStep:
        clip = np.asarray(self.clip_frames, dtype=np.float32)
        too_short = len(clip) < self.min_clip_frames
        self._clear_segment()
        if too_short:
            return RecorderStep(CaptureState.WAITING, discarded_too_short=True, reached_max_duration=reached_max_duration)
        return RecorderStep(CaptureState.WAITING, finalized_clip=clip, reached_max_duration=reached_max_duration)

    def _finish_sentence(self) -> RecorderStep:
        sentence_clip = np.asarray(self.sentence_frames, dtype=np.float32)
        finalized_clip = None
        if self.recorded_frames >= self.min_clip_frames:
            finalized_clip = np.asarray(self.clip_frames, dtype=np.float32)

        if len(sentence_clip) < self.min_clip_frames:
            self.reset()
            self.sentence_end_reported = True
            return RecorderStep(CaptureState.WAITING, discarded_too_short=len(sentence_clip) > 0, sentence_ended=True)

        self.reset()
        self.sentence_end_reported = True
        return RecorderStep(
            CaptureState.WAITING,
            finalized_clip=finalized_clip,
            finalized_sentence_clip=sentence_clip,
            sentence_ended=True,
        )

    def _clear_segment(self) -> None:
        self.state = CaptureState.WAITING
        self.pause_counter = 0
        self.observed_recording_frames = 0
        self.current_movement = 0.0
        self.clip_frames.clear()


@dataclass
class SegmentPrediction:
    """Classification result for one provisional segment."""

    sequence: np.ndarray
    result: SentenceResult
    confidence: float


@dataclass
class SegmentDecision:
    """Decision after comparing individual and concatenated segment predictions."""

    accepted: SegmentPrediction | None = None
    individual_confidence: float | None = None
    concatenated_confidence: float | None = None


@dataclass
class SegmentPredictionBuffer:
    """Hold low-confidence segments until the following segment can be compared."""

    config: LiveRecognitionConfig
    pending: SegmentPrediction | None = None
    detected_segments: int = 0
    last_individual_confidence: float | None = None
    last_concatenated_confidence: float | None = None

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    def submit(self, sequence: np.ndarray, recognizer: ContinuousRecognizer) -> SegmentDecision:
        self.detected_segments += 1
        individual = classify_segment(sequence, recognizer)
        concatenated: SegmentPrediction | None = None
        if self.pending is not None:
            concatenated_sequence = np.concatenate([self.pending.sequence, sequence], axis=0)
            concatenated = classify_segment(concatenated_sequence, recognizer)

        accepted: SegmentPrediction | None = None
        if concatenated is not None and self._is_acceptable_join(concatenated, individual):
            accepted = concatenated
            self.pending = None
        elif individual.confidence >= self.config.min_confidence:
            accepted = individual
            self.pending = None
        else:
            self.pending = individual

        self.last_individual_confidence = individual.confidence
        self.last_concatenated_confidence = None if concatenated is None else concatenated.confidence

        return SegmentDecision(
            accepted=accepted,
            individual_confidence=self.last_individual_confidence,
            concatenated_confidence=self.last_concatenated_confidence,
        )

    def _is_acceptable_join(self, joined: SegmentPrediction, individual: SegmentPrediction) -> bool:
        if self.pending is None:
            return False
        return (
            joined.confidence >= self.config.min_confidence
            and joined.confidence > self.pending.confidence
            and joined.confidence > individual.confidence
        )

    def reset_sentence(self) -> None:
        self.pending = None
        self.detected_segments = 0
        self.last_individual_confidence = None
        self.last_concatenated_confidence = None


def classify_segment(sequence: np.ndarray, recognizer: ContinuousRecognizer) -> SegmentPrediction:
    """Classify one provisional segment with the existing model feature pipeline."""
    sequence = np.asarray(sequence, dtype=np.float32)
    feature = extract_landmark_features(sequence)
    probs = recognizer.model.predict(feature[None, ...], verbose=0)[0]
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    word = recognizer.index_to_label.get(index, f"Clase {index}")
    detection = SignDetection(
        word=word,
        confidence=confidence,
        start_frame=0,
        end_frame=len(sequence),
        support=1,
        feature=feature,
        prototype_score=recognizer.prototypes.score(word, feature) if recognizer.prototypes else None,
    )
    result = SentenceResult(
        sentence=word.upper(),
        words=(word,),
        detections=(detection,),
        candidates=(detection,),
        visual_score=confidence,
        language_score=0.0,
        total_score=confidence,
    )
    return SegmentPrediction(sequence=sequence, result=result, confidence=confidence)


def parse_args() -> argparse.Namespace:
    """Parse command line options for the main recognizer."""
    parser = argparse.ArgumentParser(description="Reconoce oraciones LESCO desde cámara o un clip .npy.")
    parser.add_argument("--input-npy", type=Path, help="Clip de landmarks con shape (frames, 21, 3).")
    parser.add_argument(
        "--record-seconds",
        type=float,
        help="Herramienta debug: usa este valor como duración máxima de clip.",
    )
    parser.add_argument("--stride", type=int, help="Override debug para stride de ventanas temporales.")
    parser.add_argument("--min-confidence", type=float, help="Override debug para confianza mínima.")
    parser.add_argument("--save-clip", type=Path, help="Guarda clips capturados en .npy.")
    parser.add_argument("--debug", action="store_true", help="Muestra overlay técnico y salida detallada.")
    parser.add_argument("--config", action="store_true", help="Abre la configuración local y sale.")
    parser.add_argument("--no-prototypes", action="store_true", help="Desactiva validación gestual por prototipos.")
    return parser.parse_args()


def apply_arg_overrides(config: LiveRecognitionConfig, args: argparse.Namespace) -> LiveRecognitionConfig:
    """Apply explicit CLI overrides without changing the saved config."""
    data = config.to_dict()
    if args.stride is not None:
        data["stride"] = args.stride
    if args.min_confidence is not None:
        data["min_confidence"] = args.min_confidence
    if args.record_seconds is not None:
        data["max_clip_seconds"] = args.record_seconds
    if args.save_clip is not None:
        data["save_debug_clips"] = True
        data["save_clip_dir"] = str(args.save_clip.parent)
    if args.debug:
        data["debug"] = True
    if args.no_prototypes:
        data["use_prototypes"] = False
    return LiveRecognitionConfig.from_dict(data)


def first_hand_landmarks(tracker: HandTracker, results: object) -> np.ndarray | None:
    """Return the first detected hand, or ``None`` when tracking is empty."""
    landmarks = tracker.get_normalized_landmarks(results)
    if not landmarks:
        return None
    first_hand = np.asarray(landmarks[0], dtype=np.float32)
    if first_hand.shape != (21, 3):
        return None
    return first_hand


def save_clip_if_needed(sequence: np.ndarray, config: LiveRecognitionConfig, explicit_path: Path | None) -> None:
    """Persist a debug clip when enabled."""
    if explicit_path is not None:
        explicit_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(explicit_path, sequence)
        return
    if not config.save_debug_clips:
        return
    output_dir = Path(config.save_clip_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"clip_{int(time.time())}.npy", sequence)


def write_godot_output(output_text_path: Path, result: SentenceResult | None, status: str = "") -> None:
    """Write the final sentence and optional status for Godot."""
    if result is None:
        output_text_path.write_text(f"Oración: \nEstado: {status}", encoding="utf-8")
        return

    lines = [
        f"Oración: {result.sentence}",
        f"Score visual: {result.visual_score:.3f}",
        f"Score lenguaje: {result.language_score:.3f}",
        "Detecciones:",
    ]
    for detection in result.detections:
        lines.append(
            f"{detection.word.upper()} "
            f"conf={detection.confidence:.3f} "
            f"frames={detection.start_frame}-{detection.end_frame} "
            f"support={detection.support}"
        )
    output_text_path.write_text("\n".join(lines), encoding="utf-8")


def print_result(result: SentenceResult, debug: bool = False, processing_ms: float | None = None) -> None:
    """Print the continuous-recognition result."""
    print(f"Oración: {result.sentence}")
    if not debug:
        return

    if processing_ms is not None:
        print(f"Procesamiento: {processing_ms:.1f} ms")
    print(f"Score visual: {result.visual_score:.3f}")
    print(f"Score lenguaje: {result.language_score:.3f}")
    print("Detecciones:")
    for detection in result.detections:
        print(
            f"  {detection.word.upper()} conf={detection.confidence:.3f} "
            f"frames={detection.start_frame}-{detection.end_frame} support={detection.support}"
        )
    print("Candidatos:")
    for candidate in result.candidates:
        proto = "" if candidate.prototype_score is None else f", proto={candidate.prototype_score:.3f}"
        print(
            f"  {candidate.word.upper()} conf={candidate.confidence:.3f} "
            f"frames={candidate.start_frame}-{candidate.end_frame} support={candidate.support}{proto}"
        )


def process_sequence(
    sequence: np.ndarray,
    recognizer: ContinuousRecognizer,
    config: LiveRecognitionConfig,
) -> tuple[SentenceResult, float, int]:
    """Run continuous recognition and return result, time and window count."""
    window_count = len(sliding_window_ranges(len(sequence), stride=config.stride))
    start = time.perf_counter()
    result = recognizer.recognize(
        sequence,
        stride=config.stride,
        min_confidence=config.min_confidence,
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
    print_result(result, debug=config.debug, processing_ms=elapsed_ms)


def draw_frame(
    frame: np.ndarray,
    recorder: LandmarkClipRecorder,
    config: LiveRecognitionConfig,
    result: SentenceResult | None,
    window_count: int,
    processing_ms: float | None,
    segment_buffer: SegmentPredictionBuffer | None = None,
    sentence_status: str = "",
) -> np.ndarray:
    """Draw either normal or debug camera information."""
    if config.debug:
        return draw_debug_overlay(
            frame,
            state=recorder.state.value,
            recorded_frames=recorder.recorded_frames,
            pause_counter=recorder.pause_counter,
            end_threshold=recorder.end_threshold,
            movement=recorder.current_movement,
            no_hand_frames=recorder.no_hand_frames,
            no_hand_threshold=recorder.no_hand_threshold,
            has_pending_low_confidence=segment_buffer.has_pending if segment_buffer is not None else False,
            detected_segments=0 if segment_buffer is None else segment_buffer.detected_segments,
            individual_confidence=None if segment_buffer is None else segment_buffer.last_individual_confidence,
            concatenated_confidence=None if segment_buffer is None else segment_buffer.last_concatenated_confidence,
            sentence_status=sentence_status,
            config=config,
            window_count=window_count,
            processing_ms=processing_ms,
            result=result,
        )
    sentence = "" if result is None else result.sentence
    return draw_normal_overlay(frame, recorder.state.value, sentence=sentence)


def run_camera(args: argparse.Namespace, config: LiveRecognitionConfig, output_text_path: Path, output_frame_path: Path) -> None:
    """Run the camera state machine until the user quits."""
    tracker = HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError("No se pudo abrir la cámara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0 or fps > 120.0:
        fps = DEFAULT_FPS

    prototypes = PrototypeLibrary.from_dataset() if config.use_prototypes else None
    recognizer = ContinuousRecognizer(prototypes=prototypes)
    recorder = LandmarkClipRecorder(config=config, fps=fps)
    segment_buffer = SegmentPredictionBuffer(config=config)
    last_result: SentenceResult | None = None
    last_processing_ms: float | None = None
    last_window_count = 0
    last_sentence_status = ""
    write_godot_output(output_text_path, None, status=CaptureState.WAITING.value)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process_frame(frame)
            landmarks = first_hand_landmarks(tracker, results)
            if config.show_landmarks:
                frame = tracker.draw_landmarks(frame, results)

            if landmarks is not None and last_sentence_status:
                last_result = None
                last_sentence_status = ""
                write_godot_output(output_text_path, None, status=CaptureState.WAITING.value)

            step = recorder.step(landmarks)
            if step.discarded_too_short:
                last_result = None
                write_godot_output(output_text_path, None, status="Clip descartado por duración corta")

            if step.finalized_clip is not None:
                frame = draw_frame(
                    frame,
                    recorder,
                    config,
                    last_result,
                    last_window_count,
                    last_processing_ms,
                    segment_buffer,
                    last_sentence_status,
                )
                cv2.imshow("LESCO-AI | Reconocimiento continuo", frame)
                cv2.imwrite(str(output_frame_path), frame)
                cv2.waitKey(1)

                save_clip_if_needed(step.finalized_clip, config, args.save_clip)
                start = time.perf_counter()
                decision = segment_buffer.submit(step.finalized_clip, recognizer)
                last_processing_ms = (time.perf_counter() - start) * 1000.0
                last_window_count = 1
                if config.debug:
                    print(
                        "Segmento: "
                        f"individual={decision.individual_confidence:.3f} "
                        f"concat={'-' if decision.concatenated_confidence is None else f'{decision.concatenated_confidence:.3f}'} "
                        f"pendiente={'si' if segment_buffer.has_pending else 'no'}"
                    )

            if step.sentence_ended:
                if step.finalized_sentence_clip is not None:
                    last_result, last_processing_ms, last_window_count = process_sequence(
                        step.finalized_sentence_clip,
                        recognizer,
                        config,
                    )
                    if last_result.sentence:
                        write_godot_output(output_text_path, last_result)
                        print_result(last_result, debug=config.debug, processing_ms=last_processing_ms)
                    else:
                        last_result = None
                        write_godot_output(output_text_path, None, status="Oracion cerrada sin prediccion confiable")
                else:
                    last_result = None
                    write_godot_output(output_text_path, None, status="Oracion cerrada sin prediccion confiable")
                last_sentence_status = "Oracion cerrada por ausencia de manos"
                if config.debug:
                    print(last_sentence_status)
                segment_buffer.reset_sentence()
                recorder.reset()

            frame = draw_frame(
                frame,
                recorder,
                config,
                last_result,
                last_window_count,
                last_processing_ms,
                segment_buffer,
                last_sentence_status,
            )
            cv2.imshow("LESCO-AI | Reconocimiento continuo", frame)
            cv2.imwrite(str(output_frame_path), frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and recorder.state == CaptureState.WAITING:
                open_config_editor(default_config_path(Path(__file__).resolve().parent.parent))
                config = load_runtime_config(default_config_path(Path(__file__).resolve().parent.parent))
                recorder.config = apply_arg_overrides(config, args)
                segment_buffer.config = recorder.config
    finally:
        tracker.close()
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

    try:
        if args.input_npy is not None:
            run_offline_clip(args, config, output_text_path)
        else:
            run_camera(args, config, output_text_path, output_frame_path)
    except Exception as exc:
        output_text_path.write_text(f"Oración: \nError: {exc}", encoding="utf-8")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
