"""Utilidades para cargar dataset de señas LESCO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from feature_extraction import (
    FEATURE_PIPELINE_NAME,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    extract_landmark_features,
)


def get_project_root() -> Path:
    """Retorna la raíz del proyecto (`lesco-ai`)."""
    return Path(__file__).resolve().parent.parent


def get_default_dataset_dir() -> Path:
    """Retorna el directorio por defecto del dataset."""
    return get_project_root() / "dataset"


def get_default_models_dir() -> Path:
    """Retorna el directorio por defecto de modelos."""
    return get_project_root() / "models"


def list_label_dirs(dataset_dir: Path) -> List[Path]:
    """Lista carpetas de labels válidas dentro del dataset indicado."""
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir()])


def prepare_sample(sequence: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Convierte landmarks crudos en features finales del modelo."""
    return extract_landmark_features(sequence, target_len=target_len)


def save_label_map(label_to_index: Dict[str, int], models_dir: Path | None = None) -> Path:
    """Guarda el mapeo de labels en `models/label_map.json`."""
    if models_dir is None:
        models_dir = get_default_models_dir()

    models_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = models_dir / "label_map.json"

    index_to_label = {str(index): label for label, index in label_to_index.items()}
    payload = {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": FEATURE_SIZE,
        "feature_pipeline": FEATURE_PIPELINE_NAME,
    }

    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return label_map_path


def load_dataset(
    dataset_dir: Path | None = None,
    target_len: int = SEQUENCE_LENGTH,
    save_map: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Carga samples .npy, normaliza y retorna X, y, label_map.

    X: (n_samples, target_len, FEATURE_SIZE)
    y: (n_samples,)
    label_map: {label: class_index}
    """
    if dataset_dir is None:
        dataset_dir = get_default_dataset_dir()

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe directorio de dataset: {dataset_dir}")

    raw_label_dirs = list_label_dirs(dataset_dir)
    if not raw_label_dirs:
        raise ValueError(f"No hay carpetas de labels dentro de: {dataset_dir}")

    label_dirs: List[Path] = []
    for label_dir in raw_label_dirs:
        if any(label_dir.glob("sample_*.npy")):
            label_dirs.append(label_dir)
        else:
            print(f"[WARNING] Label sin samples, se omite: {label_dir.name}")

    if not label_dirs:
        raise ValueError("No se encontraron samples .npy en ninguna carpeta de label.")

    label_to_index = {label_dir.name: i for i, label_dir in enumerate(label_dirs)}

    samples_x: List[np.ndarray] = []
    samples_y: List[int] = []
    skipped = 0

    for label_dir in label_dirs:
        label = label_dir.name
        class_index = label_to_index[label]
        sample_files = sorted(label_dir.glob("sample_*.npy"))

        for sample_file in sample_files:
            try:
                sequence = np.load(sample_file)
                features = prepare_sample(sequence, target_len=target_len)
                samples_x.append(features)
                samples_y.append(class_index)
            except Exception as exc:
                skipped += 1
                print(f"[WARNING] Se omitió {sample_file.name} ({label}): {exc}")

    if not samples_x:
        raise ValueError(f"No se cargaron samples válidos. Revisa el contenido de {dataset_dir}.")

    X = np.array(samples_x, dtype=np.float32)
    y = np.array(samples_y, dtype=np.int64)

    if save_map:
        label_map_path = save_label_map(label_to_index)
        print(f"[INFO] label_map guardado en: {label_map_path}")

    print(
        "[INFO] Dataset cargado: "
        f"samples={len(X)}, clases={len(label_to_index)}, omitidos={skipped}, "
        f"shape_X={X.shape}"
    )

    return X, y, label_to_index
