"""Dataset-backed prototypes used for visual sign validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from src.data.dataset import get_default_dataset_dir, get_default_models_dir
from src.recognition.grouping import EPSILON
from src.vision.features import (
    FEATURE_PIPELINE_NAME,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    STATIC_SIGNATURE_DOMINANCE_THRESHOLD,
    extract_landmark_features,
    static_landmark_signature,
)

PROTOTYPE_CACHE_FILENAME = "prototype_library.npz"
PROTOTYPE_CACHE_VERSION = 1
PROTOTYPE_RADIUS_PERCENTILE = 75
PROTOTYPE_STATIC_WINDOW_FRAMES = 5


class PrototypeLibrary:
    """Class prototypes used for final visual validation."""

    def __init__(
        self,
        prototypes: dict[str, np.ndarray],
        radii: dict[str, float],
        static_prototypes: dict[str, np.ndarray] | None = None,
        static_radii: dict[str, float] | None = None,
    ) -> None:
        self.prototypes = prototypes
        self.radii = radii
        self.static_prototypes = static_prototypes if static_prototypes is not None else {}
        self.static_radii = static_radii if static_radii is not None else {}

    @classmethod
    def from_dataset(
        cls,
        dataset_dir: Path | None = None,
        cache_path: Path | None = None,
    ) -> "PrototypeLibrary":
        if dataset_dir is None:
            dataset_dir = get_default_dataset_dir()
        dataset_dir = Path(dataset_dir)
        if cache_path is None:
            cache_path = get_default_models_dir() / PROTOTYPE_CACHE_FILENAME

        sample_files = cls._sample_files(dataset_dir)
        metadata = cls._cache_metadata(dataset_dir, sample_files)
        cached = cls._load_cache(Path(cache_path), metadata)
        if cached is not None:
            return cached

        library = cls._build(dataset_dir)
        cls._save_cache(Path(cache_path), metadata, library)
        return library

    @classmethod
    def _build(cls, dataset_dir: Path) -> "PrototypeLibrary":
        prototypes: dict[str, np.ndarray] = {}
        radii: dict[str, float] = {}
        static_prototypes: dict[str, np.ndarray] = {}
        static_radii: dict[str, float] = {}
        for label_dir in sorted(dataset_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            features = []
            static_features = []
            for sample_file in sorted(label_dir.glob("sample_*.npy")):
                sample = np.load(sample_file)
                features.append(extract_landmark_features(sample))
                static_window = sample[-min(PROTOTYPE_STATIC_WINDOW_FRAMES, len(sample)) :]
                signature = static_landmark_signature(static_window)
                if bool(signature["accepted"]):
                    static_features.append(np.asarray(signature["vector"], dtype=np.float32))
            if not features:
                continue

            stacked = np.asarray(features, dtype=np.float32)
            prototype = np.mean(stacked, axis=0).astype(np.float32)
            distances = np.linalg.norm((stacked - prototype).reshape(len(stacked), -1), axis=1)
            prototypes[label_dir.name] = prototype
            radii[label_dir.name] = max(
                float(np.percentile(distances, PROTOTYPE_RADIUS_PERCENTILE)), EPSILON
            )

            if static_features:
                static_stacked = np.asarray(static_features, dtype=np.float32)
                static_prototype = np.mean(static_stacked, axis=0).astype(np.float32)
                static_distances = np.linalg.norm(static_stacked - static_prototype, axis=1)
                static_prototypes[label_dir.name] = static_prototype
                static_radii[label_dir.name] = max(
                    float(np.percentile(static_distances, PROTOTYPE_RADIUS_PERCENTILE)), EPSILON
                )

        return cls(
            prototypes=prototypes,
            radii=radii,
            static_prototypes=static_prototypes,
            static_radii=static_radii,
        )

    @staticmethod
    def _sample_files(dataset_dir: Path) -> list[Path]:
        return sorted(
            sample_file
            for label_dir in dataset_dir.iterdir()
            if label_dir.is_dir()
            for sample_file in label_dir.glob("sample_*.npy")
        )

    @staticmethod
    def _cache_metadata(dataset_dir: Path, sample_files: Sequence[Path]) -> dict[str, object]:
        digest = hashlib.sha256()
        for sample_file in sample_files:
            relative_path = sample_file.relative_to(dataset_dir).as_posix().encode("utf-8")
            digest.update(len(relative_path).to_bytes(4, "big"))
            digest.update(relative_path)
            with sample_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return {
            "cache_version": PROTOTYPE_CACHE_VERSION,
            "dataset_sha256": digest.hexdigest(),
            "sample_count": len(sample_files),
            "feature_pipeline": FEATURE_PIPELINE_NAME,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_size": FEATURE_SIZE,
            "static_signature_dominance_threshold": STATIC_SIGNATURE_DOMINANCE_THRESHOLD,
            "static_window_frames": PROTOTYPE_STATIC_WINDOW_FRAMES,
            "radius_percentile": PROTOTYPE_RADIUS_PERCENTILE,
            "epsilon": EPSILON,
            "numpy_version": np.__version__,
        }

    @classmethod
    def _load_cache(
        cls,
        cache_path: Path,
        expected_metadata: dict[str, object],
    ) -> "PrototypeLibrary | None":
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata"].item()))
                if metadata != expected_metadata:
                    return None
                labels = cached["labels"].tolist()
                static_labels = cached["static_labels"].tolist()
                prototypes_array = cached["prototypes"]
                radii_array = cached["radii"]
                static_prototypes_array = cached["static_prototypes"]
                static_radii_array = cached["static_radii"]
                if len(labels) != len(prototypes_array) or len(labels) != len(radii_array):
                    return None
                if (
                    len(static_labels) != len(static_prototypes_array)
                    or len(static_labels) != len(static_radii_array)
                ):
                    return None
                return cls(
                    prototypes={label: prototypes_array[index] for index, label in enumerate(labels)},
                    radii={label: float(radii_array[index]) for index, label in enumerate(labels)},
                    static_prototypes={
                        label: static_prototypes_array[index]
                        for index, label in enumerate(static_labels)
                    },
                    static_radii={
                        label: float(static_radii_array[index])
                        for index, label in enumerate(static_labels)
                    },
                )
        except Exception:  # La caché es opcional; cualquier daño fuerza reconstrucción.
            return None

    @staticmethod
    def _save_cache(
        cache_path: Path,
        metadata: dict[str, object],
        library: "PrototypeLibrary",
    ) -> None:
        temporary_path: Path | None = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            labels = sorted(library.prototypes)
            static_labels = sorted(library.static_prototypes)
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    labels=np.asarray(labels),
                    prototypes=np.asarray(
                        [library.prototypes[label] for label in labels], dtype=np.float32
                    ),
                    radii=np.asarray([library.radii[label] for label in labels], dtype=np.float64),
                    static_labels=np.asarray(static_labels),
                    static_prototypes=np.asarray(
                        [library.static_prototypes[label] for label in static_labels], dtype=np.float32
                    ),
                    static_radii=np.asarray(
                        [library.static_radii[label] for label in static_labels], dtype=np.float64
                    ),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, cache_path)
        except Exception:  # La aplicación debe funcionar aunque la caché no pueda guardarse.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def score(self, word: str, feature: np.ndarray) -> float:
        """Return a bounded visual compatibility score in ``(0, 1]``."""
        prototype = self.prototypes.get(word)
        if prototype is None:
            return 0.0
        radius = self.radii[word]
        distance = float(np.linalg.norm((feature - prototype).reshape(-1)))
        return float(np.exp(-distance / (radius + EPSILON)))

    def static_score(self, word: str, signature: dict[str, object] | None) -> float | None:
        """Return static landmark compatibility, or ``None`` when unavailable."""
        if signature is None or not bool(signature.get("accepted", False)):
            return None
        prototype = self.static_prototypes.get(word)
        if prototype is None:
            return None
        radius = self.static_radii[word]
        vector = np.asarray(signature["vector"], dtype=np.float32)
        distance = float(np.linalg.norm(vector - prototype))
        return float(np.exp(-distance / (radius + EPSILON)))
