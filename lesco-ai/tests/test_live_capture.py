"""Tests for live camera segmentation and runtime configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import predict_live  # noqa: E402
from feature_extraction import extract_landmark_features, static_landmark_signature, temporal_resample  # noqa: E402
from hand_tracker import select_continuous_hand, select_two_hand_slots  # noqa: E402
from model_utils import load_label_map, load_sign_model  # noqa: E402
from continuous_recognition import SentenceResult, SignDetection  # noqa: E402
from predict_live import (  # noqa: E402
    CaptureState,
    LandmarkClipRecorder,
    SegmentPrediction,
    SegmentPredictionBuffer,
    write_debug_response,
    write_godot_output,
)
from runtime_config import LiveRecognitionConfig, load_runtime_config, save_runtime_config  # noqa: E402


def hand_frame(value: float = 1.0) -> np.ndarray:
    """Return a valid synthetic hand landmark frame."""
    frame = np.zeros((21, 3), dtype=np.float32)
    frame[:, 0] = np.linspace(0.0, value, 21, dtype=np.float32)
    frame[:, 1] = np.linspace(value, 0.0, 21, dtype=np.float32)
    frame[:, 2] = value
    return frame


def moved_hand_frame(dx: float = 0.0, value: float = 1.0) -> np.ndarray:
    """Return a valid frame translated in camera space."""
    frame = hand_frame(value)
    frame[:, 0] += dx
    return frame


class ContinuousHandSelectionTests(unittest.TestCase):
    def test_uses_first_valid_hand_without_previous_reference(self) -> None:
        first = moved_hand_frame(0.0)
        second = moved_hand_frame(1.0)
        selected = select_continuous_hand([first.tolist(), second.tolist()])
        np.testing.assert_allclose(selected, first)

    def test_keeps_hand_closest_to_previous_reference(self) -> None:
        previous = moved_hand_frame(0.0)
        same_hand = moved_hand_frame(0.02)
        other_hand = moved_hand_frame(1.0)
        selected = select_continuous_hand([other_hand.tolist(), same_hand.tolist()], previous_hand=previous)
        np.testing.assert_allclose(selected, same_hand)

    def test_two_hand_slots_start_sorted_by_x(self) -> None:
        left = moved_hand_frame(0.0)
        right = moved_hand_frame(1.0)
        selected = select_two_hand_slots([right.tolist(), left.tolist()])
        np.testing.assert_allclose(selected[0], left)
        np.testing.assert_allclose(selected[1], right)

    def test_two_hand_slots_keep_previous_slot_order(self) -> None:
        previous = select_two_hand_slots([moved_hand_frame(0.0).tolist(), moved_hand_frame(1.0).tolist()])
        selected = select_two_hand_slots(
            [moved_hand_frame(1.02).tolist(), moved_hand_frame(0.02).tolist()],
            previous_frame=previous,
        )
        np.testing.assert_allclose(selected[0], moved_hand_frame(0.02))
        np.testing.assert_allclose(selected[1], moved_hand_frame(1.02))


class LiveClipRecorderTests(unittest.TestCase):
    def make_config(self, **overrides: object) -> LiveRecognitionConfig:
        data = LiveRecognitionConfig(
            movement_threshold=0.01,
            pause_frames=3,
            no_hands_timeout_seconds=0.3,
            min_clip_seconds=0.1,
            max_clip_seconds=2.0,
        ).to_dict()
        data.update(overrides)
        return LiveRecognitionConfig.from_dict(data)

    def test_does_not_start_without_motion(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(hand_frame())
        recorder.step(hand_frame())
        self.assertEqual(recorder.state, CaptureState.WAITING)
        self.assertEqual(recorder.recorded_frames, 0)

    def test_starts_when_motion_crosses_threshold(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        step = recorder.step(moved_hand_frame(0.05))
        self.assertEqual(step.state, CaptureState.MOVING)
        self.assertEqual(recorder.recorded_frames, 1)

    def test_cancels_possible_pause_when_motion_returns_before_threshold(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        recorder.step(moved_hand_frame(0.05))
        self.assertEqual(recorder.state, CaptureState.POSSIBLE_PAUSE)
        self.assertEqual(recorder.pause_counter, 1)
        self.assertEqual(len(recorder.possible_pause_frames), 1)
        recorder.step(moved_hand_frame(0.12))
        self.assertEqual(recorder.state, CaptureState.MOVING)
        self.assertEqual(recorder.pause_counter, 0)
        self.assertEqual(len(recorder.possible_pause_frames), 0)
        self.assertEqual(recorder.recorded_frames, 3)

    def test_confirms_pause_without_finishing_segment(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        step = None
        for _ in range(3):
            step = recorder.step(moved_hand_frame(0.05))
        self.assertIsNotNone(step)
        self.assertEqual(step.state, CaptureState.WAITING)
        self.assertIsNone(step.finalized_clip)
        self.assertEqual(recorder.recorded_frames, 4)

    def test_finishes_previous_segment_when_motion_returns_after_confirmed_pause(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        for _ in range(3):
            recorder.step(moved_hand_frame(0.05))
        step = recorder.step(moved_hand_frame(0.12))
        self.assertEqual(step.state, CaptureState.MOVING)
        self.assertIsNotNone(step.finalized_clip)
        self.assertEqual(step.finalized_clip.shape[0], 1)
        self.assertEqual(recorder.recorded_frames, 1)
        self.assertIsNone(step.finalized_static_signature)

    def test_finalized_segment_trims_confirmed_pause_frames(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        for _ in range(3):
            recorder.step(moved_hand_frame(0.05))

        step = recorder.step(moved_hand_frame(0.12))

        self.assertIsNotNone(step.finalized_clip)
        self.assertEqual(step.finalized_clip.shape[0], 1)
        self.assertEqual(step.finalized_start_frame, 1)
        self.assertEqual(step.finalized_end_frame, 2)
        self.assertEqual(recorder.recorded_frames, 1)

    def test_short_three_frame_sign_closed_by_pause_is_valid(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        recorder.step(moved_hand_frame(0.10))
        recorder.step(moved_hand_frame(0.15))
        for _ in range(3):
            recorder.step(moved_hand_frame(0.15))

        step = recorder.step(moved_hand_frame(0.25))

        self.assertIsNotNone(step.finalized_clip)
        self.assertFalse(step.movement_exit)
        self.assertEqual(step.finalized_clip.shape[0], 3)
        self.assertEqual(step.finalized_start_frame, 1)
        self.assertEqual(step.finalized_end_frame, 4)

    def test_discards_too_short_clip(self) -> None:
        config = self.make_config(min_clip_seconds=1.0)
        recorder = LandmarkClipRecorder(config, fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        step = None
        for _ in range(3):
            step = recorder.step(moved_hand_frame(0.05))
        self.assertIsNotNone(step)
        self.assertFalse(step.discarded_too_short)
        step = recorder.step(moved_hand_frame(0.12))
        self.assertIsNotNone(step)
        self.assertTrue(step.discarded_too_short)
        self.assertIsNone(step.finalized_clip)
        self.assertEqual(recorder.state, CaptureState.MOVING)

    def test_finishes_at_max_duration(self) -> None:
        config = self.make_config(max_clip_seconds=1.0)
        recorder = LandmarkClipRecorder(config, fps=3)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        recorder.step(moved_hand_frame(0.10))
        step = recorder.step(moved_hand_frame(0.15))
        self.assertTrue(step.reached_max_duration)
        self.assertIsNotNone(step.finalized_clip)
        self.assertEqual(step.state, CaptureState.WAITING)

    def test_confirmed_hand_absence_closes_open_segment_and_sentence(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        step = None
        for _ in range(3):
            step = recorder.step(None)
        self.assertIsNotNone(step)
        self.assertTrue(step.sentence_ended)
        self.assertIsNotNone(step.finalized_clip)
        self.assertEqual(step.finalized_clip.shape[0], 1)
        self.assertIsNotNone(step.finalized_sentence_clip)
        self.assertEqual(step.finalized_sentence_clip.shape[0], 2)
        self.assertEqual(recorder.state, CaptureState.WAITING)

    def test_pause_segment_keeps_full_sentence_clip_until_hand_absence(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        segment_step = None
        for _ in range(3):
            segment_step = recorder.step(moved_hand_frame(0.05))
        self.assertIsNotNone(segment_step)
        self.assertIsNone(segment_step.finalized_clip)
        self.assertIsNone(segment_step.finalized_sentence_clip)
        self.assertEqual(recorder.sentence_recorded_frames, 5)

        sentence_step = None
        for _ in range(3):
            sentence_step = recorder.step(None)
        self.assertIsNotNone(sentence_step)
        self.assertTrue(sentence_step.sentence_ended)
        self.assertIsNotNone(sentence_step.finalized_sentence_clip)
        self.assertEqual(sentence_step.finalized_sentence_clip.shape[0], 5)

    def test_brief_hand_absence_does_not_close_sentence(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        recorder.step(None)
        step = recorder.step(moved_hand_frame(0.10))
        self.assertFalse(step.sentence_ended)
        self.assertEqual(recorder.state, CaptureState.MOVING)
        self.assertEqual(recorder.no_hand_frames, 0)

    def test_segment_buffer_tracks_accepted_segments(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config())
        self.assertEqual(buffer.accepted_segments, [])

    def test_segment_buffer_does_not_concat_low_confidence_segments(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config())
        first_sequence = np.zeros((4, 2, 21, 3), dtype=np.float32)
        second_sequence = np.ones((5, 2, 21, 3), dtype=np.float32)

        def prediction(sequence: np.ndarray, _recognizer: object, static_signature: object | None = None) -> SegmentPrediction:
            confidence = 0.1 if len(sequence) == len(first_sequence) else 0.9
            detection = SignDetection(
                word="hola",
                confidence=confidence,
                start_frame=0,
                end_frame=len(sequence),
                support=1,
                feature=np.zeros((30, 1), dtype=np.float32),
            )
            result = SentenceResult(
                sentence="HOLA",
                words=("hola",),
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
                model_confidence=confidence,
            )

        with patch.object(predict_live, "classify_segment", side_effect=prediction) as classify:
            first_decision = buffer.submit(first_sequence, recognizer=object())
            second_decision = buffer.submit(second_sequence, recognizer=object())

        self.assertIsNone(first_decision.accepted)
        self.assertIs(second_decision.accepted.sequence, second_sequence)
        self.assertEqual(classify.call_count, 2)
        self.assertIs(classify.call_args_list[0].args[0], first_sequence)
        self.assertIs(classify.call_args_list[1].args[0], second_sequence)

    def test_segments_a_pause_b_stay_independent_and_ordered(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.5))
        submitted_sequences: list[np.ndarray] = []

        expected_a = np.asarray(
            [
                moved_hand_frame(0.05),
                moved_hand_frame(0.10),
            ],
            dtype=np.float32,
        )
        expected_b = np.asarray(
            [
                moved_hand_frame(0.20),
                moved_hand_frame(0.25),
                moved_hand_frame(0.30),
            ],
            dtype=np.float32,
        )

        def prediction(sequence: np.ndarray, _recognizer: object, static_signature: object | None = None) -> SegmentPrediction:
            submitted_sequences.append(sequence.copy())
            word = "hola" if len(submitted_sequences) == 1 else "agua"
            confidence = 0.9
            detection = SignDetection(
                word=word,
                confidence=confidence,
                start_frame=0,
                end_frame=len(sequence),
                support=1,
                feature=np.zeros((30, 1), dtype=np.float32),
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
                model_confidence=confidence,
            )

        with patch.object(predict_live, "classify_segment", side_effect=prediction) as classify:
            recorder.step(moved_hand_frame(0.0))
            recorder.step(moved_hand_frame(0.05))
            recorder.step(moved_hand_frame(0.10))
            for _ in range(3):
                recorder.step(moved_hand_frame(0.10))

            first_cut = recorder.step(moved_hand_frame(0.20))
            self.assertIsNotNone(first_cut.finalized_clip)
            buffer.submit(
                first_cut.finalized_clip,
                recognizer=object(),
                start_frame=first_cut.finalized_start_frame,
                end_frame=first_cut.finalized_end_frame,
            )

            recorder.step(moved_hand_frame(0.25))
            recorder.step(moved_hand_frame(0.30))
            for _ in range(3):
                recorder.step(moved_hand_frame(0.30))

            sentence_end = None
            for _ in range(3):
                sentence_end = recorder.step(None)

            self.assertIsNotNone(sentence_end)
            self.assertTrue(sentence_end.sentence_ended)
            self.assertIsNotNone(sentence_end.finalized_clip)
            buffer.submit(
                sentence_end.finalized_clip,
                recognizer=object(),
                start_frame=sentence_end.finalized_start_frame,
                end_frame=sentence_end.finalized_end_frame,
            )

        self.assertEqual(classify.call_count, 2)
        np.testing.assert_allclose(first_cut.finalized_clip[:, 0], expected_a)
        np.testing.assert_allclose(sentence_end.finalized_clip[:, 0], expected_b)
        np.testing.assert_allclose(submitted_sequences[0][:, 0], expected_a)
        np.testing.assert_allclose(submitted_sequences[1][:, 0], expected_b)
        self.assertEqual([len(sequence) for sequence in submitted_sequences], [2, 3])
        self.assertEqual([segment.result.words[0] for segment in buffer.accepted_segments], ["hola", "agua"])

        result = buffer.build_sentence_result()

        self.assertEqual(result.words, ("hola", "agua"))
        self.assertEqual(result.sentence, "HOLA AGUA")
        self.assertEqual([(det.word, det.start_frame, det.end_frame) for det in result.detections], [("hola", 1, 3), ("agua", 6, 9)])

    def test_exit_movement_after_confirmed_pause_is_debugged_but_not_accepted(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.5))

        def prediction(sequence: np.ndarray, _recognizer: object, static_signature: object | None = None) -> SegmentPrediction:
            word = "casa" if len(buffer.raw_debug_entries) == 0 else "dormir"
            confidence = 0.9
            detection = SignDetection(
                word=word,
                confidence=confidence,
                start_frame=0,
                end_frame=len(sequence),
                support=1,
                feature=np.zeros((30, 1), dtype=np.float32),
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
                model_confidence=confidence,
                raw_top_predictions=(("dormir", 0.42), ("casa", 0.38), ("querer", 0.09)),
            )

        with patch.object(predict_live, "classify_segment", side_effect=prediction):
            recorder.step(moved_hand_frame(0.0))
            recorder.step(moved_hand_frame(0.05))
            recorder.step(moved_hand_frame(0.10))
            for _ in range(3):
                recorder.step(moved_hand_frame(0.10))

            valid_cut = recorder.step(moved_hand_frame(0.20))
            self.assertIsNotNone(valid_cut.finalized_clip)
            self.assertFalse(valid_cut.movement_exit)
            buffer.submit(
                valid_cut.finalized_clip,
                recognizer=object(),
                start_frame=valid_cut.finalized_start_frame,
                end_frame=valid_cut.finalized_end_frame,
                movement_exit=valid_cut.movement_exit,
            )

            recorder.step(moved_hand_frame(0.25))
            exit_step = None
            for _ in range(3):
                exit_step = recorder.step(None)

            self.assertIsNotNone(exit_step)
            self.assertTrue(exit_step.sentence_ended)
            self.assertTrue(exit_step.movement_exit)
            self.assertIsNotNone(exit_step.finalized_clip)
            buffer.submit(
                exit_step.finalized_clip,
                recognizer=object(),
                start_frame=exit_step.finalized_start_frame,
                end_frame=exit_step.finalized_end_frame,
                movement_exit=exit_step.movement_exit,
            )

        self.assertEqual([segment.result.words[0] for segment in buffer.accepted_segments], ["casa"])
        self.assertEqual(len(buffer.raw_debug_entries), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug_response.txt"
            write_debug_response(path, buffer, buffer.build_sentence_result())
            output = path.read_text(encoding="utf-8")

        self.assertIn("DECISION: DESCARTADO | MOVIMIENTO DE SALIDA", output)
        self.assertIn("1. DORMIR: 0.42", output)
        self.assertIn("=== PALABRAS ACEPTADAS ===\nCASA", output)
        self.assertIn("=== SALIDA FINAL ===\nCASA", output)

    def test_single_sign_ending_by_absence_is_not_exit_movement(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        recorder.step(moved_hand_frame(0.10))

        step = None
        for _ in range(3):
            step = recorder.step(None)

        self.assertIsNotNone(step)
        self.assertTrue(step.sentence_ended)
        self.assertFalse(step.movement_exit)
        self.assertIsNotNone(step.finalized_clip)

    def test_yo_tener_casa_exit_movement_does_not_contaminate_final_sentence(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.5))
        classified_sequences: list[np.ndarray] = []
        words = ["yo", "tener", "casa", "dormir"]

        def prediction(sequence: np.ndarray, _recognizer: object, static_signature: object | None = None) -> SegmentPrediction:
            classified_sequences.append(sequence.copy())
            word = words[len(classified_sequences) - 1]
            confidence = 0.9
            detection = SignDetection(
                word=word,
                confidence=confidence,
                start_frame=0,
                end_frame=len(sequence),
                support=1,
                feature=np.zeros((30, 1), dtype=np.float32),
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
                model_confidence=confidence,
                raw_top_predictions=((word, confidence), ("casa", 0.05), ("querer", 0.03)),
            )

        def active_segment(*offsets: float) -> None:
            for offset in offsets:
                recorder.step(moved_hand_frame(offset))
            for _ in range(3):
                recorder.step(moved_hand_frame(offsets[-1]))

        with patch.object(predict_live, "classify_segment", side_effect=prediction):
            recorder.step(moved_hand_frame(0.0))
            active_segment(0.05, 0.10, 0.15)
            tener_cut = recorder.step(moved_hand_frame(0.30))
            self.assertIsNotNone(tener_cut.finalized_clip)
            buffer.submit(
                tener_cut.finalized_clip,
                recognizer=object(),
                start_frame=tener_cut.finalized_start_frame,
                end_frame=tener_cut.finalized_end_frame,
                movement_exit=tener_cut.movement_exit,
            )

            active_segment(0.35, 0.40)
            casa_cut = recorder.step(moved_hand_frame(0.55))
            self.assertIsNotNone(casa_cut.finalized_clip)
            buffer.submit(
                casa_cut.finalized_clip,
                recognizer=object(),
                start_frame=casa_cut.finalized_start_frame,
                end_frame=casa_cut.finalized_end_frame,
                movement_exit=casa_cut.movement_exit,
            )

            active_segment(0.60, 0.65)
            exit_start_cut = recorder.step(moved_hand_frame(0.85))
            self.assertIsNotNone(exit_start_cut.finalized_clip)
            buffer.submit(
                exit_start_cut.finalized_clip,
                recognizer=object(),
                start_frame=exit_start_cut.finalized_start_frame,
                end_frame=exit_start_cut.finalized_end_frame,
                movement_exit=exit_start_cut.movement_exit,
            )

            recorder.step(moved_hand_frame(0.90))
            exit_step = None
            for _ in range(3):
                exit_step = recorder.step(None)

            self.assertIsNotNone(exit_step)
            self.assertTrue(exit_step.movement_exit)
            self.assertIsNotNone(exit_step.finalized_clip)
            buffer.submit(
                exit_step.finalized_clip,
                recognizer=object(),
                start_frame=exit_step.finalized_start_frame,
                end_frame=exit_step.finalized_end_frame,
                movement_exit=exit_step.movement_exit,
            )

        result = buffer.build_sentence_result()

        self.assertEqual([segment.result.words[0] for segment in buffer.accepted_segments], ["yo", "tener", "casa"])
        self.assertEqual(result.words, ("yo", "tener", "casa"))
        self.assertEqual(result.sentence, "YO TENER CASA")
        self.assertEqual(len(classified_sequences), 4)
        self.assertEqual([len(segment.sequence) for segment in buffer.accepted_segments], [3, 3, 3])
        self.assertEqual(len(classified_sequences[-1]), 2)
        self.assertTrue(buffer.raw_debug_entries[-1].movement_exit)

    def test_segment_buffer_builds_sentence_from_accepted_segments(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config())
        feature = np.zeros((30, 1), dtype=np.float32)
        for word in ("hola", "agua"):
            detection = SignDetection(
                word=word,
                confidence=0.9,
                start_frame=0,
                end_frame=4,
                support=1,
                feature=feature,
            )
            result = SentenceResult(
                sentence=word.upper(),
                words=(word,),
                detections=(detection,),
                candidates=(detection,),
                visual_score=0.9,
                language_score=0.0,
                total_score=0.9,
            )
            buffer.accepted_segments.append(
                SegmentPrediction(
                    sequence=np.zeros((4, 2, 21, 3), dtype=np.float32),
                    result=result,
                    confidence=0.9,
                    model_confidence=0.9,
                )
            )

        result = buffer.build_sentence_result()
        self.assertEqual(result.sentence, "HOLA AGUA")
        self.assertEqual(result.words, ("hola", "agua"))
        self.assertEqual(len(result.detections), 2)

    def test_live_sentence_builder_keeps_low_confidence_yo_tener_casa(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.7))
        feature = np.zeros((30, 1), dtype=np.float32)
        for word, confidence, start, end in (
            ("yo", 0.75, 1, 3),
            ("tener", 0.92, 6, 9),
            ("casa", 0.91, 13, 16),
        ):
            detection = SignDetection(
                word=word,
                confidence=confidence,
                start_frame=0,
                end_frame=end - start,
                support=1,
                feature=feature,
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
            buffer.accepted_segments.append(
                SegmentPrediction(
                    sequence=np.zeros((end - start, 2, 21, 3), dtype=np.float32),
                    result=result,
                    confidence=confidence,
                    model_confidence=confidence,
                    start_frame=start,
                    end_frame=end,
                )
            )

        result = buffer.build_sentence_result()

        self.assertEqual(result.words, ("yo", "tener", "casa"))
        self.assertEqual(result.sentence, "YO TENER CASA")

    def test_live_sentence_builder_keeps_low_confidence_yo_querer_casa(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.7))
        feature = np.zeros((30, 1), dtype=np.float32)
        for word, confidence, start, end in (
            ("yo", 0.75, 1, 3),
            ("querer", 0.92, 6, 9),
            ("casa", 0.91, 13, 16),
        ):
            detection = SignDetection(
                word=word,
                confidence=confidence,
                start_frame=0,
                end_frame=end - start,
                support=1,
                feature=feature,
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
            buffer.accepted_segments.append(
                SegmentPrediction(
                    sequence=np.zeros((end - start, 2, 21, 3), dtype=np.float32),
                    result=result,
                    confidence=confidence,
                    model_confidence=confidence,
                    start_frame=start,
                    end_frame=end,
                )
            )

        result = buffer.build_sentence_result()

        self.assertEqual(result.words, ("yo", "querer", "casa"))
        self.assertEqual(result.sentence, "YO QUERER CASA")

    def test_debug_response_writes_raw_top_predictions_for_rejected_segments(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.probabilities = [
                    np.asarray([0.67, 0.18, 0.06, 0.05], dtype=np.float32),
                    np.asarray([0.02, 0.03, 0.91, 0.04], dtype=np.float32),
                ]
                self.calls = 0

            def predict(self, _features: np.ndarray, verbose: int = 0) -> np.ndarray:
                probabilities = self.probabilities[self.calls]
                self.calls += 1
                return probabilities[None, :]

        class FakeRecognizer:
            model = FakeModel()
            index_to_label = {0: "yo", 1: "usted", 2: "tener", 3: "casa"}
            prototypes = None

        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.70))
        first = np.zeros((24, 2, 21, 3), dtype=np.float32)
        second = np.ones((31, 2, 21, 3), dtype=np.float32)

        first_decision = buffer.submit(first, FakeRecognizer(), start_frame=0, end_frame=24)
        second_decision = buffer.submit(second, FakeRecognizer(), start_frame=30, end_frame=61)

        self.assertIsNone(first_decision.accepted)
        self.assertIsNotNone(second_decision.accepted)
        self.assertEqual(len(buffer.raw_debug_entries), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug_response.txt"
            write_debug_response(path, buffer, buffer.build_sentence_result())
            output = path.read_text(encoding="utf-8")

        self.assertIn("=== PREDICCIONES CRUDAS ===", output)
        self.assertIn("SEGMENTO 1 | FRAMES: 24 | RANGO: 0-24", output)
        self.assertIn("1. YO: 0.67", output)
        self.assertIn("2. USTED: 0.18", output)
        self.assertIn("3. TENER: 0.06", output)
        self.assertIn("DECISION: RECHAZADO | UMBRAL: 0.70", output)
        self.assertIn("SEGMENTO 2 | FRAMES: 31 | RANGO: 30-61", output)
        self.assertIn("1. TENER: 0.91", output)
        self.assertIn("DECISION: ACEPTADO", output)
        self.assertIn("=== PALABRAS ACEPTADAS ===\nTENER", output)
        self.assertIn("=== SALIDA FINAL ===\nTENER", output)

    def test_debug_response_clears_for_new_sentence_and_restarts_segment_counter(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.probabilities = [
                    np.asarray([0.91, 0.05, 0.04], dtype=np.float32),
                    np.asarray([0.03, 0.94, 0.03], dtype=np.float32),
                ]
                self.calls = 0

            def predict(self, _features: np.ndarray, verbose: int = 0) -> np.ndarray:
                probabilities = self.probabilities[self.calls]
                self.calls += 1
                return probabilities[None, :]

        class FakeRecognizer:
            model = FakeModel()
            index_to_label = {0: "casa", 1: "yo", 2: "tener"}
            prototypes = None

        buffer = SegmentPredictionBuffer(self.make_config(min_confidence=0.70))
        buffer.submit(np.zeros((24, 2, 21, 3), dtype=np.float32), FakeRecognizer(), start_frame=0, end_frame=24)
        previous_result = buffer.build_sentence_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug_response.txt"
            write_debug_response(path, buffer, previous_result)
            previous_output = path.read_text(encoding="utf-8")

            buffer.reset_debug_sentence()
            write_debug_response(path, buffer, None)
            cleared_output = path.read_text(encoding="utf-8")

            buffer.submit(np.ones((18, 2, 21, 3), dtype=np.float32), FakeRecognizer(), start_frame=0, end_frame=18)
            write_debug_response(path, buffer, buffer.build_sentence_result())
            next_output = path.read_text(encoding="utf-8")

        self.assertIn("SEGMENTO 1 | FRAMES: 24", previous_output)
        self.assertIn("CASA", previous_output)
        self.assertNotIn("SEGMENTO 1 | FRAMES: 24", cleared_output)
        self.assertIn("=== PALABRAS ACEPTADAS ===", cleared_output)
        self.assertIn("=== SALIDA FINAL ===", cleared_output)
        self.assertNotIn("CASA", cleared_output)
        self.assertIn("SEGMENTO 1 | FRAMES: 18 | RANGO: 0-18", next_output)
        self.assertIn("1. YO: 0.94", next_output)
        self.assertNotIn("=== PALABRAS ACEPTADAS ===\nCASA", next_output)
        self.assertNotIn("=== SALIDA FINAL ===\nCASA", next_output)

    def test_segment_buffer_collapses_consecutive_duplicate_words(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config())
        feature = np.zeros((30, 1), dtype=np.float32)
        for index, word in enumerate(("usted", "usted", "tener", "casa", "casa")):
            detection = SignDetection(
                word=word,
                confidence=0.8 + index * 0.01,
                start_frame=0,
                end_frame=4,
                support=1,
                feature=feature,
            )
            result = SentenceResult(
                sentence=word.upper(),
                words=(word,),
                detections=(detection,),
                candidates=(detection,),
                visual_score=detection.confidence,
                language_score=0.0,
                total_score=detection.confidence,
            )
            buffer.accepted_segments.append(
                SegmentPrediction(
                    sequence=np.zeros((4, 2, 21, 3), dtype=np.float32),
                    result=result,
                    confidence=detection.confidence,
                    model_confidence=detection.confidence,
                )
            )

        result = buffer.build_sentence_result()
        self.assertEqual(result.sentence, "USTED TENER CASA")
        self.assertEqual(result.words, ("usted", "tener", "casa"))
        self.assertEqual([detection.word for detection in result.detections], ["usted", "tener", "casa"])

    def test_godot_output_keeps_only_sentence_visual_score_and_detections(self) -> None:
        detection = SignDetection(
            word="casa",
            confidence=0.9,
            start_frame=0,
            end_frame=4,
            support=1,
            feature=np.zeros((30, 1), dtype=np.float32),
        )
        result = SentenceResult(
            sentence="CASA",
            words=("casa",),
            detections=(detection,),
            candidates=(detection,),
            visual_score=0.9,
            language_score=0.4,
            total_score=1.3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.txt"
            write_godot_output(path, result)
            output = path.read_text(encoding="utf-8")

        self.assertIn("Oración: CASA", output)
        self.assertIn("Score visual: 0.900", output)
        self.assertIn("Detecciones:", output)
        self.assertNotIn("Score lenguaje", output)

    def test_static_signature_reports_dominance_state(self) -> None:
        signature = static_landmark_signature(np.asarray([hand_frame(), hand_frame()], dtype=np.float32))
        self.assertGreaterEqual(signature["dominant_landmark"], 0)
        self.assertLess(signature["dominant_landmark"], 21)
        self.assertGreaterEqual(signature["dominance"], 0.0)
        self.assertIn(signature["accepted"], (True, False))
        self.assertIn(signature["ambiguous"], (True, False))


class RuntimeConfigTests(unittest.TestCase):
    def test_load_and_save_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.json"
            config = LiveRecognitionConfig(min_confidence=0.6, stride=3)
            save_runtime_config(config, path)
            loaded = load_runtime_config(path)
            self.assertEqual(loaded.min_confidence, 0.6)
            self.assertEqual(loaded.stride, 3)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            LiveRecognitionConfig(min_confidence=1.5).validate()
        with self.assertRaises(ValueError):
            LiveRecognitionConfig(stride=0).validate()
        with self.assertRaises(ValueError):
            LiveRecognitionConfig(min_clip_seconds=5.0, max_clip_seconds=2.0).validate()
        with self.assertRaises(ValueError):
            LiveRecognitionConfig(pause_frames=0).validate()


class DurationNormalizationRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_path = PROJECT_ROOT / "models" / "lesco_landmark_lstm.keras"
        if not model_path.exists():
            raise unittest.SkipTest("Modelo entrenado no disponible para pruebas de duración.")
        try:
            cls.model = load_sign_model(model_path)
        except ValueError as exc:
            raise unittest.SkipTest(f"Modelo entrenado incompatible con pipeline actual: {exc}") from exc
        cls.index_to_label = load_label_map(PROJECT_ROOT / "models" / "label_map.json")
        cls.sample = np.load(PROJECT_ROOT / "dataset" / "hola" / "sample_001.npy")

    def predict_label(self, sample: np.ndarray) -> str:
        features = extract_landmark_features(sample)[None, ...]
        probs = self.model.predict(features, verbose=0)[0]
        return self.index_to_label[int(np.argmax(probs))]

    def test_accelerated_sample_remains_recognizable(self) -> None:
        accelerated = temporal_resample(self.sample, target_len=15)
        self.assertEqual(self.predict_label(accelerated), "hola")

    def test_slowed_sample_remains_recognizable(self) -> None:
        slowed = temporal_resample(self.sample, target_len=60)
        self.assertEqual(self.predict_label(slowed), "hola")


if __name__ == "__main__":
    unittest.main()
