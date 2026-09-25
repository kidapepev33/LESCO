"""Tests for landmark sample recording helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.record_sign import (  # noqa: E402
    COUNTDOWN_SECONDS,
    FRAMES_PER_SAMPLE,
    HANDS_PER_FRAME,
    LANDMARK_DIMS,
    LANDMARKS_PER_HAND,
    RecordingCycle,
    RecordingState,
    fill_missing_two_hand_frames,
    parse_args,
    sample_has_two_hands,
    save_sample,
)


def one_hand_frame(value: float = 1.0) -> np.ndarray:
    frame = np.zeros((HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS), dtype=np.float32)
    frame[0, :, 0] = np.linspace(0.0, value, LANDMARKS_PER_HAND, dtype=np.float32)
    return frame


def two_hand_frame(value: float = 1.0) -> np.ndarray:
    frame = one_hand_frame(value)
    frame[1, :, 1] = np.linspace(value, 0.0, LANDMARKS_PER_HAND, dtype=np.float32)
    return frame


class RecordSignTests(unittest.TestCase):
    def test_recording_cycle_counts_down_and_captures_exactly_twenty_frames(self) -> None:
        cycle = RecordingCycle()
        cycle.start_countdown(10.0)

        self.assertEqual(cycle.state, RecordingState.COUNTDOWN)
        self.assertEqual(cycle.countdown_value(10.0), 3)
        self.assertEqual(cycle.countdown_value(11.1), 2)
        self.assertEqual(cycle.countdown_value(12.1), 1)

        cycle.update_countdown(10.0 + COUNTDOWN_SECONDS)
        self.assertEqual(cycle.state, RecordingState.CAPTURING)
        for _ in range(FRAMES_PER_SAMPLE - 1):
            self.assertFalse(cycle.add_valid_frame(one_hand_frame()))
        self.assertTrue(cycle.add_valid_frame(one_hand_frame()))
        self.assertEqual(len(cycle.frames), FRAMES_PER_SAMPLE)

    def test_recording_cycle_ignores_invalid_frames(self) -> None:
        cycle = RecordingCycle()
        cycle.start_countdown(0.0)
        cycle.update_countdown(COUNTDOWN_SECONDS)

        self.assertFalse(cycle.add_valid_frame(np.zeros((HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS))))
        self.assertEqual(len(cycle.frames), 0)

    def test_completed_cycle_saves_exactly_twenty_frames(self) -> None:
        cycle = RecordingCycle()
        cycle.start_countdown(0.0)
        cycle.update_countdown(COUNTDOWN_SECONDS)
        for _ in range(FRAMES_PER_SAMPLE):
            completed = cycle.add_valid_frame(one_hand_frame())

        self.assertTrue(completed)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_sample(Path(tmpdir), 1, cycle.frames)
            saved = np.load(output)

        self.assertEqual(saved.shape, (FRAMES_PER_SAMPLE, HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS))

    def test_stop_cancels_countdown(self) -> None:
        cycle = RecordingCycle()
        cycle.start_countdown(0.0)
        cycle.stop()

        self.assertEqual(cycle.state, RecordingState.IDLE)
        self.assertIsNone(cycle.countdown_deadline)
        cycle.update_countdown(COUNTDOWN_SECONDS)
        self.assertEqual(cycle.state, RecordingState.IDLE)

    def test_stop_discards_partial_capture_and_allows_restart(self) -> None:
        cycle = RecordingCycle()
        cycle.start_countdown(0.0)
        cycle.update_countdown(COUNTDOWN_SECONDS)
        cycle.add_valid_frame(one_hand_frame())
        cycle.stop()

        self.assertEqual(cycle.state, RecordingState.IDLE)
        self.assertEqual(cycle.frames, [])
        self.assertIsNone(cycle.previous_frame)

        cycle.start_countdown(20.0)
        self.assertEqual(cycle.state, RecordingState.COUNTDOWN)

    def test_two_hand_requirement_is_an_explicit_option(self) -> None:
        with patch.object(sys, "argv", ["record_sign.py", "--label", "cualquiera"]):
            default_args = parse_args()
        with patch.object(
            sys,
            "argv",
            ["record_sign.py", "--label", "cualquiera", "--require-two-hands"],
        ):
            two_hand_args = parse_args()

        self.assertFalse(default_args.require_two_hands)
        self.assertTrue(two_hand_args.require_two_hands)

    def test_save_sample_writes_two_hand_shape(self) -> None:
        frames = [one_hand_frame(), one_hand_frame()]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_sample(Path(tmpdir), 1, frames)
            saved = np.load(output)
        self.assertEqual(saved.shape, (2, HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS))

    def test_rejects_required_two_hand_sample_with_one_hand_frame(self) -> None:
        frames = [two_hand_frame()] * 20 + [one_hand_frame()] * 8
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_sample(Path(tmpdir), 1, frames, require_two_hands=True)

    def test_accepts_required_two_hand_sample_when_both_slots_are_present(self) -> None:
        frames = [two_hand_frame(), two_hand_frame()]
        sample = np.asarray(frames, dtype=np.float32)
        self.assertTrue(sample_has_two_hands(sample))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_sample(Path(tmpdir), 1, frames, require_two_hands=True)
            self.assertTrue(output.exists())

    def test_accepts_sample_with_eight_of_thirty_two_incomplete_frames(self) -> None:
        frames = [two_hand_frame(float(index + 1)) for index in range(32)]
        for index in range(8):
            frames[index] = one_hand_frame(float(index + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_sample(Path(tmpdir), 1, frames, require_two_hands=True)
            saved = np.load(output)

        self.assertEqual(saved.shape, (32, HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS))
        self.assertTrue(sample_has_two_hands(saved))

    def test_rejects_sample_with_ten_of_twenty_seven_incomplete_frames(self) -> None:
        frames = [two_hand_frame(float(index + 1)) for index in range(27)]
        for index in range(10):
            frames[index] = one_hand_frame(float(index + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "demasiados frames con menos de dos manos: 10/27"):
                save_sample(Path(tmpdir), 1, frames, require_two_hands=True)

    def test_rejects_sample_with_seventeen_of_twenty_seven_incomplete_frames(self) -> None:
        frames = [two_hand_frame(float(index + 1)) for index in range(27)]
        for index in range(17):
            frames[index] = one_hand_frame(float(index + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "demasiados frames con menos de dos manos: 17/27"):
                save_sample(Path(tmpdir), 1, frames, require_two_hands=True)

    def test_fills_missing_frames_with_valid_slot_data(self) -> None:
        frames = [two_hand_frame(float(index + 1)) for index in range(20)]
        frames[0][1] = 0.0
        frames[3][1] = 0.0
        sample = np.asarray(frames, dtype=np.float32)

        filled, filled_count = fill_missing_two_hand_frames(sample)

        self.assertEqual(filled_count, 2)
        np.testing.assert_allclose(filled[0, 1], sample[1, 1])
        np.testing.assert_allclose(filled[3, 1], sample[2, 1])
        self.assertTrue(sample_has_two_hands(filled))


if __name__ == "__main__":
    unittest.main()
