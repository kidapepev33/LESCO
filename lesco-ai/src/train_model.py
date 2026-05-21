"""Entrenamiento básico LSTM para clasificación de señas LESCO."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


from dataset_utils import FEATURE_SIZE, SEQUENCE_LENGTH, load_dataset

MIN_CLASSES_REQUIRED = 2
MIN_SAMPLES_REQUIRED = 10


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrena un modelo LSTM básico para LESCO.")
    parser.add_argument("--epochs", type=int, default=30, help="Cantidad de epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción test (default: 0.2)")
    return parser.parse_args()


def build_model(num_classes: int) -> keras.Model:
    """Construye el modelo LSTM solicitado."""
    model = keras.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
            layers.LSTM(64),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def validate_dataset(y: np.ndarray) -> None:
    """Valida que haya datos suficientes para entrenar."""
    num_samples = len(y)
    unique_classes = np.unique(y)

    if num_samples < MIN_SAMPLES_REQUIRED:
        raise ValueError(
            f"Muy pocos samples para entrenar: {num_samples}. "
            f"Se requieren al menos {MIN_SAMPLES_REQUIRED}."
        )

    if len(unique_classes) < MIN_CLASSES_REQUIRED:
        raise ValueError(
            f"Muy pocas clases para clasificación: {len(unique_classes)}. "
            "Agrega samples de al menos 2 señas diferentes."
        )

    class_counts = np.bincount(y)
    if np.any(class_counts < 2):
        raise ValueError(
            "Hay clases con menos de 2 samples. "
            "Se necesita al menos 2 por clase para train/test estratificado."
        )


def main() -> None:
    """Flujo principal de entrenamiento."""
    args = parse_args()

    print("[INFO] Cargando dataset...")
    X, y, label_map = load_dataset()

    print("[INFO] Validando dataset...")
    validate_dataset(y)

    num_classes = len(label_map)
    print(f"[INFO] Clases detectadas: {num_classes}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            "No se pudo dividir train/test de forma estratificada. "
            "Asegúrate de tener suficientes muestras por clase."
        ) from exc

    print(
        f"[INFO] Split -> train: {X_train.shape[0]} samples, "
        f"test: {X_test.shape[0]} samples"
    )

    model = build_model(num_classes=num_classes)
    model.summary()

    print("[INFO] Iniciando entrenamiento...")

    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
    )  

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[early_stop],
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test loss: {loss:.4f}")
    print(f"[RESULT] Test accuracy: {accuracy:.4f}")

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lesco_lstm.keras"

    model.save(model_path)
    print(f"[OK] Modelo guardado en: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
