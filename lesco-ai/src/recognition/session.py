"""Coordination state for a live recognition session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import time

import cv2
import numpy as np

from src.config.editor import open_config_editor
from src.config.runtime import LiveRecognitionConfig, apply_arg_overrides, default_config_path, load_runtime_config
from src.integration.sign_video_bridge import write_godot_output
from src.recognition.capture import (
    CaptureState,
    LandmarkClipRecorder,
    RecorderStep,
)
from src.recognition.continuous import ContinuousRecognizer, SentenceResult
from src.recognition.segments import SegmentPredictionBuffer
from src.diagnostics.debug_view import write_debug_response
from src.vision.hand_tracker import HandTracker, select_two_hand_slots


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
        recognizer = ContinuousRecognizer()
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

        self._handle_hands_returned(landmarks)
        step = self.recorder.step(landmarks)
        if step.finalized_clip is not None:
            frame = self._classify_finalized_segment(frame, step)
        if step.sentence_ended:
            self._close_sentence()
        return self.draw(frame)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def show_frame(self, frame: np.ndarray) -> None:
        """Muestra y comparte exactamente el mismo frame procesado."""
        cv2.imshow("LESCO-AI | Reconocimiento continuo", frame)
        shared_frame = frame.copy()
        temporary_path = self.output_frame_path.with_name(f"{self.output_frame_path.stem}.next.jpg")
        if cv2.imwrite(str(temporary_path), shared_frame):
            temporary_path.replace(self.output_frame_path)

    def publish_frame(self, frame: np.ndarray) -> None:
        """Compatibilidad para consumidores que aún publican un frame procesado."""
        self.show_frame(frame)

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

    def _handle_hands_returned(self, landmarks: np.ndarray | None) -> None:
        if self.state.last_sentence_status and landmarks is not None:
            self.state.clear_sentence_feedback()
            self.segment_buffer.reset_debug_sentence()
            write_godot_output(self.output_text_path, None, status=CaptureState.WAITING.value)
            write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)

    def _classify_finalized_segment(self, frame: np.ndarray, step: RecorderStep) -> np.ndarray:
        frame = self.draw(frame)
        self.show_frame(frame)
        cv2.waitKey(1)

        save_clip_if_needed(step.finalized_clip, self.config, self.args.save_clip)
        start = time.perf_counter()
        decision = self.segment_buffer.submit(
            step.finalized_clip,
            self.recognizer,
            static_signature=step.finalized_static_signature,
            start_frame=step.finalized_start_frame,
            end_frame=step.finalized_end_frame,
            movement_exit=step.movement_exit,
        )
        self.state.last_processing_ms = (time.perf_counter() - start) * 1000.0
        self.state.last_window_count = 1
        if decision.accepted is not None:
            self.state.last_result = self.segment_buffer.build_sentence_result(self.recognizer.builder)
            write_godot_output(self.output_text_path, self.state.last_result)
        write_debug_response(self.debug_response_path, self.segment_buffer, self.state.last_result)
        return frame

    def _close_sentence(self) -> None:
        final_result = self.segment_buffer.build_sentence_result(self.recognizer.builder)
        self.state.last_window_count = self.segment_buffer.detected_segments
        if final_result.sentence:
            self.state.last_result = final_result
            if self.result_printer is not None:
                self.result_printer(final_result, self.state.last_processing_ms)
        self.state.last_sentence_status = "Oracion cerrada por ausencia de manos"
        write_debug_response(self.debug_response_path, self.segment_buffer, final_result)
        self.segment_buffer.reset_sentence()
        self.recorder.reset()
        self.state.previous_selected_frame = None


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
