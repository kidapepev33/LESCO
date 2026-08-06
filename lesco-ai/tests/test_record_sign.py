"""Tests for landmark sample recording helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from record_sign import (  # noqa: E402
    HANDS_PER_FRAME,
    LANDMARK_DIMS,
    LANDMARKS_PER_HAND,
    fill_missing_two_hand_frames,
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
