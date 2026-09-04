"""Main entry point for continuous LESCO sentence recognition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
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
from continuous_recognition import (
    ContinuousRecognizer,
    PrototypeLibrary,
    SentenceBuilder,
    SentenceResult,
    SignDetection,
    StaticSegment,
    sliding_window_ranges,
)
from debug_view import draw_normal_overlay
from feature_extraction import extract_landmark_features, palm_scale, static_landmark_signature
from hand_tracker import HandTracker, select_two_hand_slots
from runtime_config import LiveRecognitionConfig, default_config_path, load_runtime_config

DEFAULT_FPS = 30.0
TWO_HAND_FRAME_SHAPE = (2, 21, 3)
ONE_HAND_FRAME_SHAPE = (21, 3)
USE_ACCELERATION_ACTIVITY = True
USE_STATIC_LANDMARK_SIGNATURES = False
ACCELERATION_ACTIVITY_WEIGHT = 0.5


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
    finalized_static_signature: dict[str, object] | None = None
    finalized_start_frame: int | None = None
    finalized_end_frame: int | None = None
    finalized_sentence_clip: np.ndarray | None = None
    discarded_too_short: bool = False
    reached_max_duration: bool = False
    sentence_ended: bool = False
    movement_exit: bool = False


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
    possible_pause_frames: list[np.ndarray] = field(default_factory=list)
    previous_landmarks: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    previous_movement_score: float = 0.0
    sentence_end_reported: bool = False
    segment_start_frame: int | None = None
    segment_started_after_confirmed_pause: bool = False

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
        self.possible_pause_frames.clear()
        self.previous_landmarks = None
        self.previous_velocity = None
        self.previous_movement_score = 0.0
        self.segment_start_frame = None
        self.segment_started_after_confirmed_pause = False

    def step(self, landmarks: np.ndarray | None) -> RecorderStep:
        """Advance the recorder using one camera frame."""
        has_hand = landmarks is not None
        frame = ensure_two_hand_frame(landmarks) if has_hand else None
        activity = self._activity_score(frame)
        self.current_movement = activity
        is_active = activity >= self.config.movement_threshold

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
            if self.clip_frames and is_active:
                return self._finalize(next_segment_frame=frame)
            if not is_active:
                return RecorderStep(self.state)

            self.state = CaptureState.MOVING
            self.pause_counter = 0
            self.observed_recording_frames = 1
            self.segment_start_frame = len(self.sentence_frames) - 1
            self.clip_frames = [frame]
            self.segment_started_after_confirmed_pause = False
            return RecorderStep(self.state)

        if self.state in {CaptureState.MOVING, CaptureState.POSSIBLE_PAUSE}:
            if is_active and self.state == CaptureState.POSSIBLE_PAUSE and self.pause_counter >= self.end_threshold:
                return self._finalize(next_segment_frame=frame)

            self.observed_recording_frames += 1
            self.clip_frames.append(frame)

            if is_active:
                self.state = CaptureState.MOVING
                self.pause_counter = 0
                self.possible_pause_frames.clear()
            else:
                self.state = CaptureState.POSSIBLE_PAUSE
                self.pause_counter += 1
                self.possible_pause_frames.append(frame)

            if self.observed_recording_frames >= self.max_clip_frames:
                return self._finalize(reached_max_duration=True)
            if self.pause_counter >= self.end_threshold:
                self.state = CaptureState.WAITING
            return RecorderStep(self.state)

        return RecorderStep(self.state)

    def _activity_score(self, landmarks: np.ndarray | None) -> float:
        if landmarks is None:
            return 0.0

        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            self.previous_velocity = np.zeros_like(landmarks, dtype=np.float32)
            return 0.0

        movements = []
        accelerations = []
        velocity_frame = np.zeros_like(landmarks, dtype=np.float32)
        for hand_index in range(landmarks.shape[0]):
            previous_hand = self.previous_landmarks[hand_index]
            current_hand = landmarks[hand_index]
            if not np.any(previous_hand) or not np.any(current_hand):
                continue
            scale = max((palm_scale(previous_hand) + palm_scale(current_hand)) / 2.0, 1e-6)
            velocity = ((current_hand - previous_hand) / scale).astype(np.float32)
            velocity_frame[hand_index] = velocity
            movements.append(self._mean_landmark_norm(velocity))
            if USE_ACCELERATION_ACTIVITY and self.previous_velocity is not None:
                acceleration = velocity - self.previous_velocity[hand_index]
                accelerations.append(self._mean_landmark_norm(acceleration))
        self.previous_landmarks = landmarks
        self.previous_velocity = velocity_frame
        if not movements:
            self.previous_movement_score = 0.0
            return 0.0
        movement_score = float(np.mean(movements))
        positive_acceleration = max(0.0, movement_score - self.previous_movement_score)
        self.previous_movement_score = movement_score
        if not USE_ACCELERATION_ACTIVITY or not accelerations:
            return movement_score
        acceleration_score = min(float(np.mean(accelerations)), positive_acceleration)
        return max(movement_score, acceleration_score * ACCELERATION_ACTIVITY_WEIGHT)

    @staticmethod
    def _mean_landmark_norm(values: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(values, axis=1)))

    def _finalize(
        self,
        reached_max_duration: bool = False,
        next_segment_frame: np.ndarray | None = None,
    ) -> RecorderStep:
        clip_frames = self._segment_frames_without_confirmed_pause()
        clip = np.asarray(clip_frames, dtype=np.float32)
        static_signature = self._confirmed_static_signature()
        start_frame = self.segment_start_frame
        end_frame = None if start_frame is None else start_frame + len(clip)
        too_short = len(clip) < self.min_clip_frames
        self._clear_segment()
        if next_segment_frame is not None:
            self.state = CaptureState.MOVING
            self.pause_counter = 0
            self.observed_recording_frames = 1
            self.segment_start_frame = len(self.sentence_frames) - 1
            self.clip_frames = [next_segment_frame]
            self.segment_started_after_confirmed_pause = True
        if too_short:
            return RecorderStep(self.state, discarded_too_short=True, reached_max_duration=reached_max_duration)
        return RecorderStep(
            self.state,
            finalized_clip=clip,
            finalized_static_signature=static_signature,
            finalized_start_frame=start_frame,
            finalized_end_frame=end_frame,
            reached_max_duration=reached_max_duration,
        )

    def _segment_frames_without_confirmed_pause(self) -> list[np.ndarray]:
        if self.pause_counter < self.end_threshold or not self.possible_pause_frames:
            return list(self.clip_frames)
        pause_len = min(len(self.possible_pause_frames), len(self.clip_frames))
        if pause_len == 0:
            return list(self.clip_frames)
        return list(self.clip_frames[:-pause_len])

    def _finish_sentence(self) -> RecorderStep:
        sentence_clip = np.asarray(self.sentence_frames, dtype=np.float32)
        finalized_clip = None
        static_signature = self._confirmed_static_signature()
        start_frame = self.segment_start_frame
        end_frame = None
        clip_frames = self._segment_frames_without_confirmed_pause()
        movement_exit = self.segment_started_after_confirmed_pause and self.pause_counter < self.end_threshold
        if len(clip_frames) >= self.min_clip_frames or (movement_exit and clip_frames):
            finalized_clip = np.asarray(clip_frames, dtype=np.float32)
            end_frame = None if start_frame is None else start_frame + len(finalized_clip)

        if len(sentence_clip) < self.min_clip_frames:
            self.reset()
            self.sentence_end_reported = True
            return RecorderStep(CaptureState.WAITING, discarded_too_short=len(sentence_clip) > 0, sentence_ended=True)

        self.reset()
        self.sentence_end_reported = True
        return RecorderStep(
            CaptureState.WAITING,
            finalized_clip=finalized_clip,
            finalized_static_signature=static_signature,
            finalized_start_frame=start_frame,
            finalized_end_frame=end_frame,
            finalized_sentence_clip=sentence_clip,
            sentence_ended=True,
            movement_exit=movement_exit,
        )

    def _clear_segment(self) -> None:
        self.state = CaptureState.WAITING
        self.pause_counter = 0
        self.observed_recording_frames = 0
        self.current_movement = 0.0
        self.clip_frames.clear()
        self.possible_pause_frames.clear()
        self.segment_start_frame = None
        self.segment_started_after_confirmed_pause = False

    def _confirmed_static_signature(self) -> dict[str, object] | None:
        if not USE_STATIC_LANDMARK_SIGNATURES:
            return None
        if len(self.possible_pause_frames) < self.end_threshold:
            return None
        return static_landmark_signature(np.asarray(self.possible_pause_frames, dtype=np.float32))


def ensure_two_hand_frame(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    frame = np.asarray(landmarks, dtype=np.float32)
    if frame.shape == TWO_HAND_FRAME_SHAPE:
        return frame
    if frame.shape == ONE_HAND_FRAME_SHAPE:
        two_hand = np.zeros(TWO_HAND_FRAME_SHAPE, dtype=np.float32)
        two_hand[0] = frame
        return two_hand
    raise ValueError(f"Frame de landmarks inválido. Esperado {TWO_HAND_FRAME_SHAPE}, obtenido {frame.shape}")


@dataclass
class SegmentPrediction:
    """Classification result for one provisional segment."""

    sequence: np.ndarray
    result: SentenceResult
    confidence: float
    model_confidence: float
    static_signature: dict[str, object] | None = None
    static_score: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    raw_top_predictions: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RawSegmentDebugEntry:
    """Raw model output for one classified live segment."""

    segment_number: int
    frame_count: int
    top_predictions: tuple[tuple[str, float], ...]
    accepted: bool
    threshold: float
    movement_exit: bool = False
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass
class SegmentDecision:
    """Decision for one independently classified segment."""

    accepted: SegmentPrediction | None = None
    individual_confidence: float | None = None


@dataclass
class SegmentPredictionBuffer:
    """Collect accepted independent segment predictions for the current sentence."""

    config: LiveRecognitionConfig
    accepted_segments: list[SegmentPrediction] = field(default_factory=list)
    raw_debug_entries: list[RawSegmentDebugEntry] = field(default_factory=list)
    detected_segments: int = 0
    last_individual_confidence: float | None = None
    last_static_signature: dict[str, object] | None = None
    last_static_score: float | None = None

    def submit(
        self,
        sequence: np.ndarray,
        recognizer: ContinuousRecognizer,
        static_signature: dict[str, object] | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        movement_exit: bool = False,
    ) -> SegmentDecision:
        self.detected_segments += 1
        individual = classify_segment(sequence, recognizer, static_signature=static_signature)
        individual = replace(individual, start_frame=start_frame, end_frame=end_frame)

        accepted: SegmentPrediction | None = None
        if not movement_exit and individual.confidence >= self.config.min_confidence:
            accepted = individual

        self.raw_debug_entries.append(
            RawSegmentDebugEntry(
                segment_number=self.detected_segments,
                frame_count=len(sequence),
                top_predictions=individual.raw_top_predictions,
                accepted=accepted is not None,
                threshold=self.config.min_confidence,
                movement_exit=movement_exit,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        self.raw_debug_entries = self.raw_debug_entries[-5:]

        if accepted is not None:
            self.accepted_segments.append(accepted)

        self.last_individual_confidence = individual.confidence
        self.last_static_signature = static_signature
        self.last_static_score = individual.static_score

        return SegmentDecision(
            accepted=accepted,
            individual_confidence=self.last_individual_confidence,
        )

    def reset_sentence(self) -> None:
        self.accepted_segments.clear()
        self.detected_segments = 0
        self.last_individual_confidence = None
        self.last_static_signature = None
        self.last_static_score = None

    def reset_debug_sentence(self) -> None:
        self.reset_sentence()
        self.raw_debug_entries.clear()

    def build_sentence_result(self) -> SentenceResult:
        if not self.accepted_segments:
            return SentenceResult("", (), (), (), 0.0, 0.0, 0.0)

        detections: list[SignDetection] = []
        visual_scores: list[float] = []
        frame_offset = 0
        for segment in self.accepted_segments:
            visual_scores.append(segment.confidence)
            segment_offset = segment.start_frame if segment.start_frame is not None else frame_offset
            for detection in segment.result.detections:
                detections.append(
                    SignDetection(
                        word=detection.word,
                        confidence=detection.confidence,
                        start_frame=segment_offset + detection.start_frame,
                        end_frame=segment_offset + detection.end_frame,
                        support=detection.support,
                        feature=detection.feature,
                        prototype_score=detection.prototype_score,
                        static_signature=detection.static_signature,
                        static_score=detection.static_score,
                        static_dominant_landmark=detection.static_dominant_landmark,
                        static_dominance=detection.static_dominance,
                        static_accepted=detection.static_accepted,
                    )
                )
            frame_offset += len(segment.sequence)

        result = SentenceBuilder().build(detections, suppress_competing=False)
        visual_score = float(np.mean(visual_scores)) if visual_scores else 0.0
        return SentenceResult(
            sentence=result.sentence,
            words=result.words,
            detections=result.detections,
            candidates=result.candidates,
            visual_score=visual_score,
            language_score=result.language_score,
            total_score=visual_score + result.language_score,
        )


def classify_segment(
    sequence: np.ndarray,
    recognizer: ContinuousRecognizer,
    static_signature: dict[str, object] | None = None,
) -> SegmentPrediction:
    """Classify one provisional segment with the existing model feature pipeline."""
    sequence = np.asarray(sequence, dtype=np.float32)
    feature = extract_landmark_features(sequence)
    probs = recognizer.model.predict(feature[None, ...], verbose=0)[0]
    raw_top_predictions = top_model_predictions(probs, recognizer.index_to_label, top_k=3)
    index = int(np.argmax(probs))
    model_confidence = float(probs[index])
    word = recognizer.index_to_label.get(index, f"Clase {index}")
    static_score = recognizer.prototypes.static_score(word, static_signature) if recognizer.prototypes else None
    confidence = model_confidence if static_score is None else float(0.5 * model_confidence + 0.5 * static_score)
    detection = SignDetection(
        word=word,
        confidence=confidence,
        start_frame=0,
        end_frame=len(sequence),
        support=1,
        feature=feature,
        prototype_score=recognizer.prototypes.score(word, feature) if recognizer.prototypes else None,
        static_score=static_score,
        static_dominant_landmark=None if static_signature is None else int(static_signature["dominant_landmark"]),
        static_dominance=None if static_signature is None else float(static_signature["dominance"]),
        static_accepted=None if static_signature is None else bool(static_signature["accepted"]),
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
    return SegmentPrediction(
        sequence=sequence,
        result=result,
        confidence=confidence,
        model_confidence=model_confidence,
        static_signature=static_signature,
        static_score=static_score,
        raw_top_predictions=raw_top_predictions,
    )


def top_model_predictions(
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    top_k: int = 3,
) -> tuple[tuple[str, float], ...]:
    """Return raw model top-k labels before runtime filters."""
    probs = np.asarray(probabilities, dtype=np.float32)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return tuple((index_to_label.get(int(index), f"Clase {int(index)}"), float(probs[index])) for index in top_indices)


def write_debug_response(
    debug_path: Path,
    segment_buffer: SegmentPredictionBuffer,
    final_result: SentenceResult | None = None,
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
        if entry.movement_exit:
            lines.append("DECISION: DESCARTADO | MOVIMIENTO DE SALIDA")
        elif entry.accepted:
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
    if args.no_prototypes:
        data["use_prototypes"] = False
    return LiveRecognitionConfig.from_dict(data)


def two_hand_landmarks(
    tracker: HandTracker,
    results: object,
    previous_frame: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return two stable hand slots, or ``None`` when tracking is empty."""
    landmarks = tracker.get_normalized_landmarks(results)
    frame = select_two_hand_slots(landmarks, previous_frame=previous_frame)
    if not np.any(frame):
        return None
    return frame


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
    """Draw normal camera information."""
    sentence = "" if result is None else result.sentence
    return draw_normal_overlay(frame, recorder.state.value, sentence=sentence)


def run_camera(
    args: argparse.Namespace,
    config: LiveRecognitionConfig,
    output_text_path: Path,
    output_frame_path: Path,
    debug_response_path: Path,
) -> None:
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
    sentence_static_segments: list[StaticSegment] = []
    previous_selected_frame: np.ndarray | None = None
    write_godot_output(output_text_path, None, status=CaptureState.WAITING.value)
    write_debug_response(debug_response_path, segment_buffer, last_result)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process_frame(frame)
            landmarks = two_hand_landmarks(tracker, results, previous_frame=previous_selected_frame)
            if landmarks is not None:
                previous_selected_frame = landmarks
            elif recorder.state == CaptureState.WAITING:
                previous_selected_frame = None
            if config.show_landmarks:
                frame = tracker.draw_landmarks(frame, results)

            step = recorder.step(landmarks)
            if last_sentence_status and step.state == CaptureState.MOVING and recorder.recorded_frames == 1:
                last_result = None
                last_sentence_status = ""
                last_processing_ms = None
                last_window_count = 0
                segment_buffer.reset_debug_sentence()
                write_godot_output(output_text_path, None, status=CaptureState.WAITING.value)
                write_debug_response(debug_response_path, segment_buffer, last_result)

            if step.discarded_too_short:
                last_result = None
                write_godot_output(output_text_path, None, status="Clip descartado por duración corta")

            if step.finalized_clip is not None:
                if (
                    step.finalized_static_signature is not None
                    and step.finalized_start_frame is not None
                    and step.finalized_end_frame is not None
                ):
                    sentence_static_segments.append(
                        (
                            step.finalized_start_frame,
                            step.finalized_end_frame,
                            step.finalized_static_signature,
                        )
                    )
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
                segment_buffer.submit(
                    step.finalized_clip,
                    recognizer,
                    static_signature=step.finalized_static_signature,
                    start_frame=step.finalized_start_frame,
                    end_frame=step.finalized_end_frame,
                    movement_exit=step.movement_exit,
                )
                last_processing_ms = (time.perf_counter() - start) * 1000.0
                last_window_count = 1
                write_debug_response(debug_response_path, segment_buffer, last_result)

            if step.sentence_ended:
                last_result = segment_buffer.build_sentence_result()
                last_window_count = segment_buffer.detected_segments
                if last_result.sentence:
                    write_godot_output(output_text_path, last_result)
                    print_result(last_result, processing_ms=last_processing_ms)
                else:
                    last_result = None
                    write_godot_output(output_text_path, None, status="Oracion cerrada sin prediccion confiable")
                last_sentence_status = "Oracion cerrada por ausencia de manos"
                write_debug_response(debug_response_path, segment_buffer, last_result)
                segment_buffer.reset_sentence()
                sentence_static_segments.clear()
                recorder.reset()
                previous_selected_frame = None

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
