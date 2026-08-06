"""Grabador de muestras LESCO en formato fijo de dos manos."""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from hand_tracker import HandTracker, select_two_hand_slots

HANDS_PER_FRAME = 2
LANDMARKS_PER_HAND = 21
LANDMARK_DIMS = 3
MIN_FRAMES_TO_SAVE = 10
TWO_HAND_LABELS = {"casa", "querer", "ayuda", "escuela"}
MAX_MISSING_TWO_HAND_RATIO = 0.25
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Graba muestras de señas en formato (frames, 2, 21, 3).")
    parser.add_argument(
        "--label",
        required=True,
        type=str,
        help="Nombre de la seña a grabar (ejemplo: hola)",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    """Normaliza el label para usarlo como nombre de carpeta."""
    return label.strip().lower().replace(" ", "_")


def get_label_dir(label: str) -> Path:
    """Retorna la ruta `dataset/<label>` relativa a la raíz del proyecto."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dataset" / label


def get_next_sample_index(label_dir: Path) -> int:
    """Calcula el siguiente índice de muestra basado en archivos existentes."""
    max_index = 0
    for sample_file in label_dir.glob("sample_*.npy"):
        stem = sample_file.stem  # sample_001
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        if parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))
    return max_index + 1


def save_sample(
    label_dir: Path,
    sample_index: int,
    frames: List[np.ndarray],
    require_two_hands: bool = False,
) -> Path:
    """Guarda una muestra en disco como archivo .npy con shape (frames, 2, 21, 3)."""
    sample_array = np.array(frames, dtype=np.float32)

    if sample_array.ndim != 4 or sample_array.shape[1:] != (HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS):
        raise ValueError(
            "Shape inválido. "
            f"Esperado: (frames, {HANDS_PER_FRAME}, {LANDMARKS_PER_HAND}, {LANDMARK_DIMS}), "
            f"obtenido: {sample_array.shape}"
        )
    filled_frames = 0
    if require_two_hands:
        sample_array, filled_frames = fill_missing_two_hand_frames(sample_array)

    output_path = label_dir / f"sample_{sample_index:03d}.npy"
    np.save(output_path, sample_array)
    if filled_frames:
        print(f"Muestra aceptada con {filled_frames} frames rellenados")
    return output_path


def hands_present_per_frame(sample_array: np.ndarray) -> np.ndarray:
    """Return how many hand slots contain landmarks on each frame."""
    present = np.any(np.abs(sample_array) > EPSILON, axis=(2, 3))
    return np.sum(present, axis=1)


def sample_has_two_hands(sample_array: np.ndarray) -> bool:
    """Return true when both hand slots are present in every saved frame."""
    return bool(np.all(hands_present_per_frame(sample_array) == HANDS_PER_FRAME))


def hand_slot_is_present(sample_array: np.ndarray) -> np.ndarray:
    """Return a boolean matrix indicating present hand slots per frame."""
    return np.any(np.abs(sample_array) > EPSILON, axis=(2, 3))


def fill_missing_two_hand_frames(sample_array: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill small two-hand tracking gaps using the nearest valid slot value."""
    present = hand_slot_is_present(sample_array)
    missing_frame_mask = np.sum(present, axis=1) < HANDS_PER_FRAME
    missing_frames = int(np.sum(missing_frame_mask))
    total_frames = len(sample_array)
    if missing_frames == 0:
        return sample_array, 0

    if missing_frames / max(total_frames, 1) > MAX_MISSING_TWO_HAND_RATIO:
        raise ValueError(f"demasiados frames con menos de dos manos: {missing_frames}/{total_frames}")

    filled = sample_array.copy()
    for hand_index in range(HANDS_PER_FRAME):
        valid_indices = np.flatnonzero(present[:, hand_index])
        if len(valid_indices) == 0:
            raise ValueError(f"demasiados frames con menos de dos manos: {missing_frames}/{total_frames}")

        last_valid = None
        first_valid = int(valid_indices[0])
        for frame_index in range(total_frames):
            if present[frame_index, hand_index]:
                last_valid = filled[frame_index, hand_index].copy()
                continue
            if last_valid is not None:
                filled[frame_index, hand_index] = last_valid
            else:
                filled[frame_index, hand_index] = filled[first_valid, hand_index]

    return filled, missing_frames


def draw_overlay(
    frame: np.ndarray,
    label: str,
    recorded_frames: int,
    saved_samples: int,
    is_recording: bool,
) -> np.ndarray:
    """Dibuja información de estado e instrucciones sobre el frame."""
    status_text = "REC" if is_recording else "IDLE"

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Frames grabados: {recorded_frames}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Muestras guardadas: {saved_samples}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Estado: {status_text}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 100, 255) if is_recording else (180, 180, 180),
        2,
    )
    cv2.putText(
        frame,
        "r: iniciar | s: guardar | q: salir",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


def extract_two_hands(
    landmarks: List[List[Tuple[float, float, float]]],
    previous_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Extrae dos slots de manos estables con shape (2, 21, 3)."""
    return select_two_hand_slots(landmarks, previous_frame=previous_frame)


def main() -> None:
    """Loop principal: cámara, detección y guardado de muestras por tecla."""
    args = parse_args()
    label = sanitize_label(args.label)
    requires_two_hands = label in TWO_HAND_LABELS

    label_dir = get_label_dir(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    next_sample_index = get_next_sample_index(label_dir)
    saved_samples = next_sample_index - 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        print("Revisa conexión/permisos y el CAMERA_INDEX en src/config.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    is_recording = False
    current_frames: List[np.ndarray] = []
    previous_recording_frame: np.ndarray | None = None

    print(f"Label activo: {label}")
    if requires_two_hands:
        print("[INFO] Esta clase requiere dos manos detectadas durante toda la muestra.")
    print("Presiona 'r' para iniciar una muestra.")
    print(f"Presiona 's' para guardar (mínimo {MIN_FRAMES_TO_SAVE} frames válidos).")
    print("Presiona 'q' para salir.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] No se pudo leer un frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, results)

            normalized_landmarks = tracker.get_normalized_landmarks(results)

            if is_recording:
                two_hands = extract_two_hands(normalized_landmarks, previous_frame=previous_recording_frame)
                if np.any(two_hands):
                    current_frames.append(two_hands)
                    previous_recording_frame = two_hands

            frame = draw_overlay(
                frame=frame,
                label=label,
                recorded_frames=len(current_frames),
                saved_samples=saved_samples,
                is_recording=is_recording,
            )
            cv2.imshow("LESCO-AI | Record Sign", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not is_recording:
                current_frames = []
                previous_recording_frame = None
                is_recording = True
                print("[INFO] Grabación iniciada.")
            if key == ord("s"):
                if not is_recording:
                    print("[WARNING] No hay grabación activa. Presiona 'r' para iniciar.")
                    continue

                if len(current_frames) < MIN_FRAMES_TO_SAVE:
                    print(
                        "[WARNING] Muestra descartada: "
                        f"{len(current_frames)} frames válidos "
                        f"(mínimo requerido: {MIN_FRAMES_TO_SAVE})."
                    )
                    is_recording = False
                    current_frames = []
                    previous_recording_frame = None
                    continue

                try:
                    output_file = save_sample(
                        label_dir,
                        next_sample_index,
                        current_frames,
                        require_two_hands=requires_two_hands,
                    )
                except ValueError as exc:
                    print(f"Muestra descartada: {exc}")
                    is_recording = False
                    current_frames = []
                    previous_recording_frame = None
                    continue
                saved_samples += 1
                next_sample_index += 1
                is_recording = False
                current_frames = []
                previous_recording_frame = None
                print(f"[OK] Muestra guardada: {output_file}")

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Programa finalizado.")


if __name__ == "__main__":
    main()
