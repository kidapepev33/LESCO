"""Shared model and label-map helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tensorflow import keras

from dataset_utils import FEATURE_SIZE, SEQUENCE_LENGTH, get_default_models_dir

MODEL_FILENAME = "lesco_landmark_lstm.keras"
LABEL_MAP_FILENAME = "label_map.json"


def load_label_map(label_map_path: Path | None = None) -> dict[int, str]:
    """Load ``label_map.json`` and return an index-to-label mapping."""
    if label_map_path is None:
        label_map_path = get_default_models_dir() / LABEL_MAP_FILENAME

    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "index_to_label" in data:
        return {int(k): v for k, v in data["index_to_label"].items()}

    if "label_to_index" in data:
        return {idx: label for label, idx in data["label_to_index"].items()}

    raise ValueError("Formato de label_map inválido. Falta index_to_label o label_to_index.")


def validate_model_shape(model: keras.Model) -> None:
    """Ensure the model matches the active feature pipeline."""
    input_shape = model.input_shape
    if len(input_shape) != 3 or input_shape[1:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(
            "El modelo no coincide con el pipeline actual. "
            f"Esperado: (None, {SEQUENCE_LENGTH}, {FEATURE_SIZE}), obtenido: {input_shape}. "
            "Vuelve a entrenar con: python src/train_model.py"
        )


def load_sign_model(model_path: Path | None = None) -> keras.Model:
    """Load the trained sign model."""
    if model_path is None:
        model_path = get_default_models_dir() / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo entrenado: {model_path}")

    model = keras.models.load_model(model_path)
    validate_model_shape(model)
    return model
