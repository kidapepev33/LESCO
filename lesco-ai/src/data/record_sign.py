"""Grabador de muestras LESCO en formato fijo de dos manos."""

import argparse
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import time
from typing import List, Tuple

import cv2
import numpy as np

from src.config.constants import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HANDS_PER_FRAME,
    LANDMARK_DIMS,
    LANDMARKS_PER_HAND,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.data.dataset import get_default_dataset_dir
from src.vision.hand_tracker import HandTracker, select_two_hand_slots

COUNTDOWN_SECONDS = 3.0
FRAMES_PER_SAMPLE = 20
MAX_MISSING_TWO_HAND_RATIO = 0.25
EPSILON = 1e-6


class RecordingState(str, Enum):
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    CAPTURING = "CAPTURING"


@dataclass
class RecordingCycle:
    """Controla el ciclo countdown/captura sin depender de la cámara."""

    state: RecordingState = RecordingState.IDLE
    countdown_deadline: float | None = None
    frames: List[np.ndarray] = field(default_factory=list)
    previous_frame: np.ndarray | None = None

    def start_countdown(self, now: float) -> None:
        self.state = RecordingState.COUNTDOWN
        self.countdown_deadline = now + COUNTDOWN_SECONDS
        self.frames.clear()
        self.previous_frame = None

    def stop(self) -> None:
        self.state = RecordingState.IDLE
        self.countdown_deadline = None
        self.frames.clear()
        self.previous_frame = None

    def update_countdown(self, now: float) -> None:
        if self.state == RecordingState.COUNTDOWN and now >= self.countdown_deadline:
            self.state = RecordingState.CAPTURING
            self.countdown_deadline = None

    def countdown_value(self, now: float) -> int | None:
        if self.state != RecordingState.COUNTDOWN or self.countdown_deadline is None:
            return None
        return max(1, int(math.ceil(self.countdown_deadline - now)))

    def add_valid_frame(self, frame: np.ndarray) -> bool:
        if self.state != RecordingState.CAPTURING or not np.any(frame):
            return False
        if len(self.frames) < FRAMES_PER_SAMPLE:
            self.frames.append(frame)
            self.previous_frame = frame
        return len(self.frames) == FRAMES_PER_SAMPLE


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Graba muestras de señas en formato (frames, 2, 21, 3).")
    parser.add_argument(
        "--label",
        required=True,
        type=str,
        help="Nombre de la seña a grabar (ejemplo: hola)",
    )
    parser.add_argument(
        "--require-two-hands",
        action="store_true",
        help="Exige datos válidos en ambas manos sin depender del nombre de la seña.",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    """Normaliza el label para usarlo como nombre de carpeta."""
    return label.strip().lower().replace(" ", "_")


def get_label_dir(label: str) -> Path:
    """Retorna la ruta `dataset/<label>` relativa a la raíz del proyecto."""
    return get_default_dataset_dir() / label


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
    state: RecordingState,
    countdown: int | None,
) -> np.ndarray:
    """Dibuja información de estado e instrucciones sobre el frame."""
    if state == RecordingState.COUNTDOWN:
        status_text = f"Siguiente grabacion en: {countdown}"
        status_color = (0, 220, 255)
    elif state == RecordingState.CAPTURING:
        status_text = f"Grabando: {recorded_frames} / {FRAMES_PER_SAMPLE} frames"
        status_color = (0, 100, 255)
    else:
        status_text = "Listo para grabar"
        status_color = (180, 180, 180)

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
        status_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
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
        f"Estado: {state.value}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )
    cv2.putText(
        frame,
        "r: grabar/parar | q: salir",
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
    requires_two_hands = args.require_two_hands

    label_dir = get_label_dir(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    next_sample_index = get_next_sample_index(label_dir)
    saved_samples = next_sample_index - 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        print("Revisa conexión/permisos y CAMERA_INDEX en src/config/constants.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cycle = RecordingCycle()

    print(f"Label activo: {label}")
    if requires_two_hands:
        print("[INFO] Esta clase requiere dos manos detectadas durante toda la muestra.")
    print("Presiona 'r' para iniciar o parar la grabación automática.")
    print(f"Cada muestra se guarda automáticamente al alcanzar {FRAMES_PER_SAMPLE} frames válidos.")
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
            now = time.monotonic()
            cycle.update_countdown(now)

            if cycle.state == RecordingState.CAPTURING:
                two_hands = extract_two_hands(normalized_landmarks, previous_frame=cycle.previous_frame)
                if cycle.add_valid_frame(two_hands):
                    try:
                        output_file = save_sample(
                            label_dir,
                            next_sample_index,
                            cycle.frames,
                            require_two_hands=requires_two_hands,
                        )
                    except ValueError as exc:
                        print(f"Muestra descartada: {exc}")
                    else:
                        saved_samples += 1
                        next_sample_index += 1
                        print(f"[OK] Muestra guardada: {output_file}")
                    now = time.monotonic()
                    cycle.start_countdown(now)

            frame = draw_overlay(
                frame=frame,
                label=label,
                recorded_frames=len(cycle.frames),
                saved_samples=saved_samples,
                state=cycle.state,
                countdown=cycle.countdown_value(now),
            )
            cv2.imshow("LESCO-AI | Record Sign", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if cycle.state == RecordingState.IDLE:
                    cycle.start_countdown(now)
                    print("[INFO] Ciclo de grabación iniciado.")
                else:
                    discarded_frames = len(cycle.frames)
                    cycle.stop()
                    print(f"[INFO] Grabación detenida. Frames parciales descartados: {discarded_frames}.")

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Programa finalizado.")


if __name__ == "__main__":
    main()
