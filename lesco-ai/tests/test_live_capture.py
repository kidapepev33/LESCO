"""Tests for live camera segmentation and runtime configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_extraction import extract_landmark_features, static_landmark_signature, temporal_resample  # noqa: E402
from hand_tracker import select_continuous_hand, select_two_hand_slots  # noqa: E402
from model_utils import load_label_map, load_sign_model  # noqa: E402
from predict_live import CaptureState, LandmarkClipRecorder, SegmentPredictionBuffer  # noqa: E402
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

    def test_finishes_after_three_low_activity_frames(self) -> None:
        recorder = LandmarkClipRecorder(self.make_config(), fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        step = None
        for _ in range(3):
            step = recorder.step(moved_hand_frame(0.05))
        self.assertIsNotNone(step)
        self.assertEqual(step.state, CaptureState.WAITING)
        self.assertIsNotNone(step.finalized_clip)
        self.assertEqual(step.finalized_clip.shape[0], 4)
        self.assertIsNotNone(step.finalized_static_signature)
        self.assertIn("dominant_landmark", step.finalized_static_signature)
        self.assertIn("dominance", step.finalized_static_signature)

    def test_discards_too_short_clip(self) -> None:
        config = self.make_config(min_clip_seconds=1.0)
        recorder = LandmarkClipRecorder(config, fps=10)
        recorder.step(moved_hand_frame(0.0))
        recorder.step(moved_hand_frame(0.05))
        step = None
        for _ in range(3):
            step = recorder.step(moved_hand_frame(0.05))
        self.assertIsNotNone(step)
        self.assertTrue(step.discarded_too_short)
        self.assertIsNone(step.finalized_clip)
        self.assertEqual(recorder.state, CaptureState.WAITING)

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
        self.assertIsNotNone(segment_step.finalized_clip)
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

    def test_segment_buffer_does_not_build_final_words(self) -> None:
        buffer = SegmentPredictionBuffer(self.make_config())
        self.assertFalse(hasattr(buffer, "accepted_words"))

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
            config = LiveRecognitionConfig(min_confidence=0.6, stride=3, debug=True)
            save_runtime_config(config, path)
            loaded = load_runtime_config(path)
            self.assertEqual(loaded.min_confidence, 0.6)
            self.assertEqual(loaded.stride, 3)
            self.assertTrue(loaded.debug)

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
