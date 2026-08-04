"""Predicción en vivo de señas LESCO con el pipeline robusto de landmarks."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from dataset_utils import FEATURE_SIZE, SEQUENCE_LENGTH, prepare_sample
from hand_tracker import HandTracker

CONFIDENCE_THRESHOLD = 0.70
MODEL_FILENAME = "lesco_landmark_lstm.keras"


def load_label_map(label_map_path: Path) -> dict[int, str]:
    """Carga label_map.json y devuelve índice->label."""
    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "index_to_label" in data:
        return {int(k): v for k, v in data["index_to_label"].items()}

    if "label_to_index" in data:
        return {idx: label for label, idx in data["label_to_index"].items()}

    raise ValueError("Formato de label_map inválido. Falta index_to_label o label_to_index.")


def validate_model_shape(model: keras.Model) -> None:
    """Evita ejecutar modelos antiguos con un pipeline de features distinto."""
    input_shape = model.input_shape
    if len(input_shape) != 3 or input_shape[1:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(
            "El modelo no coincide con el pipeline actual. "
            f"Esperado: (None, {SEQUENCE_LENGTH}, {FEATURE_SIZE}), obtenido: {input_shape}. "
            "Vuelve a entrenar con: python src/train_model.py"
        )


def draw_overlay(
    frame: np.ndarray,
    prediction_text: str,
    confidence_text: str,
    sequence_size: int,
) -> np.ndarray:
    """Dibuja información de predicción sobre el frame."""
    cv2.putText(
        frame,
        f"Prediccion: {prediction_text}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Confianza: {confidence_text}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Secuencia: {sequence_size}/{SEQUENCE_LENGTH}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "q: salir",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return frame


def main() -> None:
    """Loop principal para inferencia en vivo con cámara."""
    project_root = Path(__file__).resolve().parent.parent
    godot_bridge_dir = project_root / "godot_bridge"
    godot_bridge_dir.mkdir(exist_ok=True)

    output_text_path = godot_bridge_dir / "output.txt"
    output_frame_path = godot_bridge_dir / "frame.jpg"
    model_path = project_root / "models" / MODEL_FILENAME
    label_map_path = project_root / "models" / "label_map.json"

    if not model_path.exists():
        print(f"[ERROR] No existe modelo entrenado: {model_path}")
        print("Ejecuta primero: python src/train_model.py")
        return

    if not label_map_path.exists():
        print(f"[ERROR] No existe label_map: {label_map_path}")
        print("Ejecuta primero: python src/train_model.py")
        return

    print("[INFO] Cargando modelo...")
    model = keras.models.load_model(model_path)
    validate_model_shape(model)
    index_to_label = load_label_map(label_map_path)

    tracker = HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        print("Revisa conexión/permisos y el CAMERA_INDEX en src/config.py")
        tracker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    sequence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    prediction_text = "Sin prediccion"
    confidence_text = "0.00"

    print("[INFO] Iniciando predicción en vivo. Presiona 'q' para salir.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] No se pudo leer un frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, results)

            landmarks = tracker.get_normalized_landmarks(results)
            if landmarks:
                first_hand = np.array(landmarks[0], dtype=np.float32)
                if first_hand.shape == (21, 3):
                    sequence_buffer.append(first_hand)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                raw_sequence = np.array(sequence_buffer, dtype=np.float32)  # (30, 21, 3)
                model_input = prepare_sample(raw_sequence, target_len=SEQUENCE_LENGTH)
                model_input = np.expand_dims(model_input, axis=0)

                probs = model.predict(model_input, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])

                if best_conf > CONFIDENCE_THRESHOLD:
                    prediction_text = index_to_label.get(best_idx, f"Clase {best_idx}")
                    confidence_text = f"{best_conf:.2f}"

                    sequence_buffer.clear()

                else:
                    prediction_text = "Sin prediccion"
                    confidence_text = f"{best_conf:.2f}"

            frame = draw_overlay(
                frame,
                prediction_text=prediction_text,
                confidence_text=confidence_text,
                sequence_size=len(sequence_buffer),
            )

            output_text_path.write_text(
                f"Predicción: {prediction_text}\n"
                f"Confianza: {confidence_text}\n"
                f"Secuencia: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                encoding="utf-8"
            )

            cv2.imwrite(str(output_frame_path), frame)

            cv2.imshow("LESCO-AI | Predict Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Programa finalizado.")


if __name__ == "__main__":
    main()
