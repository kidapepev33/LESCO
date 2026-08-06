"""Tests for the landmark feature pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_extraction import (  # noqa: E402
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    extract_landmark_features,
    transform_sequence,
)


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        frame = rng.normal(0.0, 0.05, size=(21, 3)).astype(np.float32)
        frame[0, :] = 0.0
        frame[5, 0] += 0.25
        frame[9, 1] += 0.28
        frame[13, 0] -= 0.20
        frame[17, 1] -= 0.22

        self.static_sequence = np.repeat(frame[None, :, :], 24, axis=0)
        self.sequence = self.static_sequence.copy()

    def moving_sequence(self, axis: int, distance: float = 0.45) -> np.ndarray:
        moved = self.static_sequence.copy()
        trajectory = np.linspace(0.0, distance, num=moved.shape[0], dtype=np.float32)
        moved[:, :, axis] += trajectory[:, None]
        return moved

    def circular_sequence(self, radius: float = 0.25) -> np.ndarray:
        moved = self.static_sequence.copy()
        t = np.linspace(0.0, 2.0 * np.pi, num=moved.shape[0], dtype=np.float32)
        moved[:, :, 0] += radius * np.cos(t)[:, None]
        moved[:, :, 1] += radius * np.sin(t)[:, None]
        return moved

    def test_feature_shape(self) -> None:
        features = extract_landmark_features(self.sequence)
        self.assertEqual(features.shape, (SEQUENCE_LENGTH, FEATURE_SIZE))
        self.assertEqual(features.dtype, np.float32)

    def test_two_hand_feature_shape(self) -> None:
        two_hand = np.zeros((self.sequence.shape[0], 2, 21, 3), dtype=np.float32)
        two_hand[:, 0] = self.sequence
        two_hand[:, 1] = self.moving_sequence(axis=1)
        features = extract_landmark_features(two_hand)
        self.assertEqual(features.shape, (SEQUENCE_LENGTH, FEATURE_SIZE))
        self.assertEqual(FEATURE_SIZE, 436)

    def test_translation_invariance(self) -> None:
        sequence = self.moving_sequence(axis=0)
        original = extract_landmark_features(sequence)
        positions = [
            (-0.8, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.0, -0.6, 0.0),
            (0.0, 0.6, 0.0),
            (0.3, -0.4, 0.2),
        ]

        for offset in positions:
            moved = extract_landmark_features(transform_sequence(sequence, offset=offset))
            np.testing.assert_allclose(original, moved, rtol=1e-5, atol=1e-5)

    def test_scale_invariance(self) -> None:
        sequence = self.moving_sequence(axis=1)
        original = extract_landmark_features(sequence)
        for scale in (0.45, 0.75, 1.5, 2.25):
            scaled = extract_landmark_features(transform_sequence(sequence, scale=scale))
            np.testing.assert_allclose(original, scaled, rtol=1e-5, atol=1e-5)

    def test_translation_and_scale_invariance_together(self) -> None:
        original = extract_landmark_features(self.moving_sequence(axis=0))
        transformed = transform_sequence(self.moving_sequence(axis=0), offset=(0.5, -0.35, 0.1), scale=1.8)
        features = extract_landmark_features(transformed)
        np.testing.assert_allclose(original, features, rtol=1e-5, atol=1e-5)

    def test_static_hand_and_moving_hand_are_different(self) -> None:
        static = extract_landmark_features(self.static_sequence)
        moving = extract_landmark_features(self.moving_sequence(axis=0))
        self.assertGreater(float(np.max(np.abs(static - moving))), 0.25)

    def test_horizontal_and_vertical_motion_are_different(self) -> None:
        horizontal = extract_landmark_features(self.moving_sequence(axis=0))
        vertical = extract_landmark_features(self.moving_sequence(axis=1))
        self.assertGreater(float(np.max(np.abs(horizontal - vertical))), 0.25)

    def test_same_trajectory_from_different_absolute_positions_is_equivalent(self) -> None:
        trajectory = self.moving_sequence(axis=0)
        shifted = transform_sequence(trajectory, offset=(0.7, -0.4, 0.2))
        np.testing.assert_allclose(
            extract_landmark_features(trajectory),
            extract_landmark_features(shifted),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_circular_motion_is_not_static(self) -> None:
        static = extract_landmark_features(self.static_sequence)
        circular = extract_landmark_features(self.circular_sequence())
        self.assertGreater(float(np.max(np.abs(static - circular))), 0.25)


if __name__ == "__main__":
    unittest.main()
