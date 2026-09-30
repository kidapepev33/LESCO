"""Tests for the dataset-backed prototype cache."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition import prototypes  # noqa: E402
from src.recognition.prototypes import PrototypeLibrary  # noqa: E402


class PrototypeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.cache = self.root / "models" / "prototype_library.npz"
        self._write_sample("hola", 1, 1.0)
        self._write_sample("agua", 1, 2.0)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_sample(self, label: str, index: int, value: float) -> Path:
        path = self.dataset / label / f"sample_{index:03d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(value * 100 + index))
        sample = rng.normal(value, 0.1, size=(8, 2, 21, 3)).astype(np.float32)
        np.save(path, sample)
        return path

    def _load(self) -> PrototypeLibrary:
        return PrototypeLibrary.from_dataset(self.dataset, cache_path=self.cache)

    def assert_libraries_equal(self, first: PrototypeLibrary, second: PrototypeLibrary) -> None:
        self.assertEqual(first.prototypes.keys(), second.prototypes.keys())
        self.assertEqual(first.radii, second.radii)
        self.assertEqual(first.static_prototypes.keys(), second.static_prototypes.keys())
        self.assertEqual(first.static_radii, second.static_radii)
        for label in first.prototypes:
            np.testing.assert_array_equal(first.prototypes[label], second.prototypes[label])
        for label in first.static_prototypes:
            np.testing.assert_array_equal(first.static_prototypes[label], second.static_prototypes[label])

    def test_cached_library_is_exactly_equivalent_and_skips_rebuild(self) -> None:
        original = self._load()
        self.assertTrue(self.cache.is_file())

        with patch.object(prototypes, "extract_landmark_features", side_effect=AssertionError("rebuild")):
            cached = self._load()

        self.assert_libraries_equal(original, cached)

    def test_added_modified_and_deleted_samples_invalidate_cache(self) -> None:
        self._load()
        changes = (
            lambda: self._write_sample("hola", 2, 3.0),
            lambda: self._write_sample("agua", 1, 4.0),
            lambda: (self.dataset / "hola" / "sample_002.npy").unlink(),
        )
        original_extractor = prototypes.extract_landmark_features
        for change in changes:
            change()
            with patch.object(prototypes, "extract_landmark_features", wraps=original_extractor) as extractor:
                self._load()
            self.assertGreater(extractor.call_count, 0)

    def test_construction_parameter_change_invalidates_cache(self) -> None:
        self._load()
        original_extractor = prototypes.extract_landmark_features
        with (
            patch.object(prototypes, "PROTOTYPE_RADIUS_PERCENTILE", 60),
            patch.object(prototypes, "extract_landmark_features", wraps=original_extractor) as extractor,
        ):
            self._load()
        self.assertGreater(extractor.call_count, 0)

    def test_corrupt_cache_is_rebuilt(self) -> None:
        self.cache.parent.mkdir(parents=True)
        self.cache.write_bytes(b"not an npz file")

        library = self._load()

        self.assertEqual(set(library.prototypes), {"agua", "hola"})
        with np.load(self.cache, allow_pickle=False) as cached:
            self.assertIn("metadata", cached.files)

    def test_save_failure_does_not_prevent_build(self) -> None:
        with patch.object(prototypes.os, "replace", side_effect=OSError("read-only")):
            library = self._load()

        self.assertEqual(set(library.prototypes), {"agua", "hola"})
        self.assertFalse(self.cache.exists())
        self.assertEqual(list(self.cache.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
