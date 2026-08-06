"""Offline tests for continuous sentence recognition."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from continuous_recognition import (  # noqa: E402
    ContinuousRecognizer,
    PrototypeLibrary,
    SentenceBuilder,
    SignDetection,
    WindowPrediction,
    concatenate_samples,
    group_repeated_detections,
)
from feature_extraction import FEATURE_SIZE, SEQUENCE_LENGTH, extract_landmark_features  # noqa: E402
from model_utils import load_label_map, load_sign_model  # noqa: E402


def synthetic_feature(value: float = 0.0) -> np.ndarray:
    return np.full((SEQUENCE_LENGTH, FEATURE_SIZE), value, dtype=np.float32)


class TemporalDeduplicationTests(unittest.TestCase):
    def test_merges_overlapping_querer_detections(self) -> None:
        feature = synthetic_feature()
        predictions = [
            WindowPrediction("querer", 0.96, 8, 58, feature),
            WindowPrediction("querer", 0.94, 32, 70, feature),
        ]
        grouped = group_repeated_detections(predictions)
        self.assertEqual([det.word for det in grouped], ["querer"])
        self.assertEqual((grouped[0].start_frame, grouped[0].end_frame), (8, 70))
        self.assertEqual(grouped[0].support, 2)

    def test_merges_overlapping_tener_detections(self) -> None:
        feature = synthetic_feature()
        predictions = [
            WindowPrediction("tener", 0.97, 0, 50, feature),
            WindowPrediction("tener", 0.95, 24, 58, feature),
        ]
        grouped = group_repeated_detections(predictions)
        self.assertEqual([det.word for det in grouped], ["tener"])
        self.assertEqual((grouped[0].start_frame, grouped[0].end_frame), (0, 58))
        self.assertEqual(grouped[0].support, 2)

    def test_keeps_separated_querer_repetition(self) -> None:
        feature = synthetic_feature()
        predictions = [
            WindowPrediction("querer", 0.96, 8, 45, feature),
            WindowPrediction("querer", 0.94, 55, 90, feature),
        ]
        grouped = group_repeated_detections(predictions)
        self.assertEqual([det.word for det in grouped], ["querer", "querer"])
        self.assertEqual([(det.start_frame, det.end_frame) for det in grouped], [(8, 45), (55, 90)])

    def test_builder_keeps_separated_agua_repetition(self) -> None:
        feature = synthetic_feature()
        detections = [
            SignDetection("agua", 0.98, 0, 30, 2, feature),
            SignDetection("agua", 0.97, 48, 78, 2, feature),
        ]
        result = SentenceBuilder().build(detections)
        self.assertEqual(list(result.words), ["agua", "agua"])

    def test_grouping_keeps_separated_agua_repetition(self) -> None:
        feature = synthetic_feature()
        predictions = [
            WindowPrediction("agua", 0.98, 0, 30, feature),
            WindowPrediction("agua", 0.97, 48, 78, feature),
        ]
        grouped = group_repeated_detections(predictions)
        self.assertEqual([det.word for det in grouped], ["agua", "agua"])
        self.assertEqual([(det.start_frame, det.end_frame) for det in grouped], [(0, 30), (48, 78)])


class ContinuousRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = PROJECT_ROOT / "dataset"
        cls.model_path = PROJECT_ROOT / "models" / "lesco_landmark_lstm.keras"
        if not cls.model_path.exists():
            raise unittest.SkipTest("Modelo entrenado no disponible para pruebas continuas.")

        cls.index_to_label = load_label_map(PROJECT_ROOT / "models" / "label_map.json")
        cls.labels = ["agua", "casa", "hola", "querer", "si", "yo"]
        cls.samples = {label: cls.load_sample(label) for label in cls.labels}
        cls.sample_features = {
            label: extract_landmark_features(sample) for label, sample in cls.samples.items()
        }
        try:
            cls.model = load_sign_model(cls.model_path)
        except ValueError as exc:
            raise unittest.SkipTest(f"Modelo entrenado incompatible con pipeline actual: {exc}") from exc
        cls.prototypes = PrototypeLibrary.from_dataset(cls.dataset_dir)

    @classmethod
    def load_sample(cls, label: str, index: int = 1) -> np.ndarray:
        return np.load(cls.dataset_dir / label / f"sample_{index:03d}.npy")

    def make_recognizer(self) -> ContinuousRecognizer:
        return ContinuousRecognizer(
            model=self.model,
            index_to_label=self.index_to_label,
            prototypes=self.prototypes,
            builder=SentenceBuilder(beam_width=6),
        )

    def recognize_labels(self, labels: list[str]) -> tuple[list[str], object]:
        clip = concatenate_samples([self.samples[label] for label in labels], transition_frames=2)
        result = self.make_recognizer().recognize(
            clip,
            window_size=24,
            stride=4,
            min_confidence=0.45,
            top_k=2,
        )
        return list(result.words), result

    def test_preserves_order_and_builds_sentence(self) -> None:
        words, result = self.recognize_labels(["hola", "agua"])
        self.assertEqual(words, ["hola", "agua"])
        self.assertEqual(result.sentence, "HOLA AGUA")
        self.assertGreaterEqual(len(result.candidates), 2)

    def test_common_phrase_yo_querer_agua(self) -> None:
        words, result = self.recognize_labels(["yo", "querer", "agua"])
        self.assertEqual(words, ["yo", "querer", "agua"])
        self.assertTrue(all(det.start_frame < det.end_frame for det in result.detections))

    def test_keeps_real_repetition(self) -> None:
        words, _ = self.recognize_labels(["hola", "hola"])
        self.assertEqual(words, ["hola", "hola"])

    def test_noise_and_transition_are_rejected(self) -> None:
        rng = np.random.default_rng(123)
        noise = rng.normal(2.0, 0.5, size=(18, 21, 3)).astype(np.float32)
        clip = concatenate_samples([noise, self.samples["hola"], self.samples["agua"]], transition_frames=6)
        result = self.make_recognizer().recognize(
            clip,
            window_size=24,
            stride=4,
            min_confidence=0.45,
            top_k=2,
        )
        self.assertEqual(list(result.words), ["hola", "agua"])

    def test_groups_overlapping_duplicate_windows(self) -> None:
        feature = self.sample_features["hola"]
        predictions = [
            WindowPrediction("hola", 0.91, 0, 24, feature),
            WindowPrediction("hola", 0.96, 4, 28, feature),
            WindowPrediction("hola", 0.90, 8, 32, feature),
        ]
        grouped = group_repeated_detections(predictions)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].word, "hola")
        self.assertEqual(grouped[0].support, 3)

    def test_builder_removes_false_transition_repeat(self) -> None:
        feature_hola = self.sample_features["hola"]
        feature_agua = self.sample_features["agua"]
        detections = [
            SignDetection("hola", 0.99, 0, 46, 4, feature_hola, 0.8),
            SignDetection("hola", 0.98, 28, 58, 2, feature_hola, 0.7),
            SignDetection("agua", 0.99, 42, 70, 3, feature_agua, 0.8),
        ]
        result = SentenceBuilder().build(detections)
        self.assertEqual(list(result.words), ["hola", "agua"])


class SentenceBuilderCleanupTests(unittest.TestCase):
    def test_removes_direct_consecutive_duplicates(self) -> None:
        feature = synthetic_feature()
        detections = [
            SignDetection("agua", 0.91, 0, 30, 2, feature),
            SignDetection("agua", 0.96, 31, 61, 3, feature),
            SignDetection("querer", 0.94, 68, 98, 2, feature),
        ]
        result = SentenceBuilder().build(detections)
        self.assertEqual(list(result.words), ["agua", "querer"])
        self.assertEqual(result.detections[0].confidence, 0.96)
        self.assertEqual(result.detections[0].support, 3)

    def test_replaces_stuck_duplicate_with_nearby_alternative(self) -> None:
        feature = synthetic_feature()
        detections = [
            SignDetection("agua", 0.97, 0, 30, 3, feature),
            SignDetection("agua", 0.96, 31, 61, 3, feature),
            SignDetection("tener", 0.91, 31, 61, 2, feature),
        ]
        result = SentenceBuilder().build(detections)
        self.assertEqual(list(result.words), ["agua", "tener"])

    def test_prefers_yo_tener_order_for_overlapping_pair(self) -> None:
        feature = synthetic_feature()
        detections = [
            SignDetection("tener", 0.997, 0, 54, 7, feature),
            SignDetection("yo", 0.942, 28, 62, 2, feature),
            SignDetection("bano", 0.961, 48, 84, 3, feature),
        ]
        result = SentenceBuilder().build(detections)
        self.assertEqual(list(result.words), ["yo", "tener", "bano"])


if __name__ == "__main__":
    unittest.main()
