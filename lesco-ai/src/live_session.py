"""Mutable live-recognition session state and recorder state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable
import time

import cv2
import numpy as np

from config_ui import open_config_editor
from continuous_recognition import ContinuousRecognizer, PrototypeLibrary, SentenceResult
from debug_view import draw_frame, write_debug_response
from feature_extraction import palm_scale, static_landmark_signature
from hand_tracker import HandTracker, select_two_hand_slots
from runtime_config import LiveRecognitionConfig, apply_arg_overrides, default_config_path, load_runtime_config
from segment_prediction import SegmentPredictionBuffer
from sign_video_bridge import write_godot_output

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


@dataclass
class LiveSessionState:
    """Mutable values shared across live frame processing."""

    last_result: SentenceResult | None = None
    last_processing_ms: float | None = None
    last_window_count: int = 0
    last_sentence_status: str = ""
    previous_selected_frame: np.ndarray | None = None

    def clear_sentence_feedback(self) -> None:
        self.last_result = None
        self.last_sentence_status = ""
        self.last_processing_ms = None
        self.last_window_count = 0


@dataclass
class LiveRecognitionSession:
    """Coordinate tracker output, recorder events, debug files and Godot output."""

    args: object
    config: LiveRecognitionConfig
    output_text_path: Path
    output_frame_path: Path
    debug_response_path: Path
    tracker: HandTracker
    recognizer: ContinuousRecognizer
    recorder: LandmarkClipRecorder
    segment_buffer: SegmentPredictionBuffer
    project_root: Path
    result_printer: Callable[[SentenceResult, float | None], None] | None = None
    state: LiveSessionState = field(default_factory=LiveSessionState)

    @classmethod
    def create(
        cls,
        args: object,
        config: LiveRecognitionConfig,
        output_text_path: Path,
        output_frame_path: Path,
        debug_response_path: Path,
        tracker: HandTracker,
        fps: float,
        project_root: Path,
        result_printer: Callable[[SentenceResult, float | None], None] | None = None,
    ) -> "LiveRecognitionSession":
        prototypes = PrototypeLibrary.from_dataset() if config.use_prototypes else None
        recognizer = ContinuousRecognizer(prototypes=prototypes)
        recorder = LandmarkClipRecorder(config=config, fps=fps)
        segment_buffer = SegmentPredictionBuffer(config=config)
        session = cls(
            args=args,
            config=config,
            output_text_path=output_text_path,
            output_frame_path=output_frame_path,
            debug_response_path=debug_response_path,
            tracker=tracker,
            recognizer=recognizer,
            recorder=recorder,
            segment_buffer=segment_buffer,
            project_root=project_root,
            result_printer=result_printer,
        )
        session.initialize_outputs()
        return session

    def initialize_outputs(self) -> None:
        write_godot_output(self.output_text_path, None, status=CaptureState.WAITING.value)
        write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.tracker.process_frame(frame)
        landmarks = two_hand_landmarks(self.tracker, results, previous_frame=self.state.previous_selected_frame)
        if landmarks is not None:
            self.state.previous_selected_frame = landmarks
        elif self.recorder.state == CaptureState.WAITING:
            self.state.previous_selected_frame = None
        if self.config.show_landmarks:
            frame = self.tracker.draw_landmarks(frame, results)

        step = self.recorder.step(landmarks)
        self._handle_new_sentence_start(step)
        self._handle_discarded_clip(step)
        if step.finalized_clip is not None:
            frame = self._classify_finalized_segment(frame, step)
        if step.sentence_ended:
            self._close_sentence()
        return self.draw(frame)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        return draw_frame(
            frame,
            self.recorder,
            self.config,
            self.state.last_result,
            self.state.last_window_count,
            self.state.last_processing_ms,
            self.segment_buffer,
            self.state.last_sentence_status,
        )

    def publish_frame(self, frame: np.ndarray) -> None:
        cv2.imshow("LESCO-AI | Reconocimiento continuo", frame)
        cv2.imwrite(str(self.output_frame_path), frame)

    def handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return False
        if key == ord("c") and self.recorder.state == CaptureState.WAITING:
            config_path = default_config_path(self.project_root)
            open_config_editor(config_path)
            config = load_runtime_config(config_path)
            self.config = apply_arg_overrides(config, self.args)
            self.recorder.config = self.config
            self.segment_buffer.config = self.config
        return True

    def _handle_new_sentence_start(self, step: RecorderStep) -> None:
        if self.state.last_sentence_status and step.state == CaptureState.MOVING and self.recorder.recorded_frames == 1:
            self.state.clear_sentence_feedback()
            self.segment_buffer.reset_debug_sentence()
            write_godot_output(self.output_text_path, None, status=CaptureState.WAITING.value)
            write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)

    def _handle_discarded_clip(self, step: RecorderStep) -> None:
        if step.discarded_too_short:
            self.state.last_result = None
            write_godot_output(self.output_text_path, None, status="Clip descartado por duración corta")

    def _classify_finalized_segment(self, frame: np.ndarray, step: RecorderStep) -> np.ndarray:
        frame = self.draw(frame)
        self.publish_frame(frame)
        cv2.waitKey(1)

        save_clip_if_needed(step.finalized_clip, self.config, self.args.save_clip)
        start = time.perf_counter()
        self.segment_buffer.submit(
            step.finalized_clip,
            self.recognizer,
            static_signature=step.finalized_static_signature,
            start_frame=step.finalized_start_frame,
            end_frame=step.finalized_end_frame,
            movement_exit=step.movement_exit,
        )
        self.state.last_processing_ms = (time.perf_counter() - start) * 1000.0
        self.state.last_window_count = 1
        write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)
        return frame

    def _close_sentence(self) -> None:
        self.state.last_result = self.segment_buffer.build_sentence_result()
        self.state.last_window_count = self.segment_buffer.detected_segments
        if self.state.last_result.sentence:
            write_godot_output(self.output_text_path, self.state.last_result)
            if self.result_printer is not None:
                self.result_printer(self.state.last_result, self.state.last_processing_ms)
        else:
            self.state.last_result = None
            write_godot_output(self.output_text_path, None, status="Oracion cerrada sin prediccion confiable")
        self.state.last_sentence_status = "Oracion cerrada por ausencia de manos"
        write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)
        self.segment_buffer.reset_sentence()
        self.recorder.reset()
        self.state.previous_selected_frame = None


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
