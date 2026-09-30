"""Landmark clip segmentation for live recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np

from src.config.constants import ONE_HAND_FRAME_SHAPE, TWO_HAND_FRAME_SHAPE
from src.config.runtime import DEFAULT_FPS, LiveRecognitionConfig
from src.vision.features import palm_scale, static_landmark_signature

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
    current_movement: float = 0.0
    clip_frames: list[np.ndarray] = field(default_factory=list)
    sentence_frames: list[np.ndarray] = field(default_factory=list)
    possible_pause_frames: list[np.ndarray] = field(default_factory=list)
    previous_landmarks: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    previous_movement_score: float = 0.0
    sentence_end_reported: bool = False
    segment_start_frame: int | None = None
    segment_started_at: float | None = None
    pause_started_at: float | None = None
    no_hands_started_at: float | None = None
    sentence_started_at: float | None = None

    @property
    def end_threshold(self) -> int:
        return self.config.pause_frames

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
        self.current_movement = 0.0
        self.clip_frames.clear()
        self.sentence_frames.clear()
        self.possible_pause_frames.clear()
        self.previous_landmarks = None
        self.previous_velocity = None
        self.previous_movement_score = 0.0
        self.segment_start_frame = None
        self.segment_started_at = None
        self.pause_started_at = None
        self.no_hands_started_at = None
        self.sentence_started_at = None

    def step(self, landmarks: np.ndarray | None, now: float | None = None) -> RecorderStep:
        """Advance the recorder using one camera frame."""
        current_time = time.monotonic() if now is None else now
        has_hand = landmarks is not None
        frame = ensure_two_hand_frame(landmarks) if has_hand else None
        activity = self._activity_score(frame)
        self.current_movement = activity
        is_active = activity >= self.config.movement_threshold

        if not has_hand:
            self.no_hand_frames += 1
            self.current_movement = 0.0
            if self.no_hands_started_at is None:
                self.no_hands_started_at = current_time
            if (
                self.no_hands_started_at is not None
                and current_time - self.no_hands_started_at + 1e-9
                >= self.config.no_hands_timeout_seconds
                and not self.sentence_end_reported
            ):
                return self._finish_sentence(current_time)
            return RecorderStep(self.state)

        self.no_hand_frames = 0
        self.no_hands_started_at = None
        self.sentence_end_reported = False
        if self.sentence_started_at is None:
            self.sentence_started_at = current_time
        self.sentence_frames.append(frame)

        if self.state == CaptureState.WAITING:
            if not is_active:
                return RecorderStep(self.state)

            self.state = CaptureState.MOVING
            self.pause_counter = 0
            self.segment_start_frame = len(self.sentence_frames) - 1
            self.segment_started_at = current_time
            self.clip_frames = [frame]
            return RecorderStep(self.state)

        if self.state in {CaptureState.MOVING, CaptureState.POSSIBLE_PAUSE}:
            self.clip_frames.append(frame)

            if is_active:
                self.state = CaptureState.MOVING
                self.pause_counter = 0
                self.possible_pause_frames.clear()
                self.pause_started_at = None
            else:
                self.state = CaptureState.POSSIBLE_PAUSE
                self.pause_counter += 1
                self.possible_pause_frames.append(frame)
                if self.pause_started_at is None:
                    self.pause_started_at = current_time

            if current_time - self.segment_started_at + 1e-9 >= self.config.max_clip_seconds:
                return self._finalize(current_time, reached_max_duration=True)
            if self.pause_counter >= self.end_threshold:
                return self._finalize(current_time)
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
        current_time: float,
        reached_max_duration: bool = False,
    ) -> RecorderStep:
        clip_frames = self._segment_frames_without_confirmed_pause()
        clip = np.asarray(clip_frames, dtype=np.float32)
        static_signature = self._confirmed_static_signature()
        start_frame = self.segment_start_frame
        end_frame = None if start_frame is None else start_frame + len(clip)
        segment_end = self.pause_started_at if self.pause_started_at is not None else current_time
        duration = 0.0 if self.segment_started_at is None else segment_end - self.segment_started_at
        too_short = duration < self.config.min_clip_seconds
        self._clear_segment()
        if too_short:
            return RecorderStep(
                self.state,
                discarded_too_short=True,
                reached_max_duration=reached_max_duration,
            )
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

    def _finish_sentence(self, current_time: float) -> RecorderStep:
        sentence_clip = np.asarray(self.sentence_frames, dtype=np.float32)
        finalized_clip = None
        static_signature = self._confirmed_static_signature()
        start_frame = self.segment_start_frame
        end_frame = None
        clip_frames = self._segment_frames_without_confirmed_pause()
        segment_end = self.no_hands_started_at if self.no_hands_started_at is not None else current_time
        segment_duration = 0.0 if self.segment_started_at is None else segment_end - self.segment_started_at
        if clip_frames and segment_duration + 1e-9 >= self.config.min_clip_seconds:
            finalized_clip = np.asarray(clip_frames, dtype=np.float32)
            end_frame = None if start_frame is None else start_frame + len(finalized_clip)

        sentence_duration = 0.0 if self.sentence_started_at is None else segment_end - self.sentence_started_at
        if sentence_duration < self.config.min_clip_seconds:
            self.reset()
            self.sentence_end_reported = True
            return RecorderStep(
                CaptureState.WAITING,
                discarded_too_short=len(sentence_clip) > 0,
                sentence_ended=True,
            )

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
            movement_exit=False,
        )

    def _clear_segment(self) -> None:
        self.state = CaptureState.WAITING
        self.pause_counter = 0
        self.current_movement = 0.0
        self.clip_frames.clear()
        self.possible_pause_frames.clear()
        self.segment_start_frame = None
        self.segment_started_at = None
        self.pause_started_at = None

    def _confirmed_static_signature(self) -> dict[str, object] | None:
        if not USE_STATIC_LANDMARK_SIGNATURES:
            return None
        if len(self.possible_pause_frames) < self.end_threshold:
            return None
        return static_landmark_signature(np.asarray(self.possible_pause_frames, dtype=np.float32))


def ensure_two_hand_frame(landmarks: np.ndarray | None) -> np.ndarray | None:
    """Normalize a landmark frame to the two-hand representation."""
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
