"""Módulo para detección y seguimiento de manos con MediaPipe."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

LANDMARKS_PER_HAND = 21
LANDMARK_DIMS = 3
HANDS_PER_FRAME = 2


class HandTracker:
    """Encapsula la lógica de MediaPipe Hands para detectar y dibujar manos."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray):
        """Procesa un frame BGR y devuelve el resultado de MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Dibuja landmarks y conexiones en el frame si hay manos detectadas."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_normalized_landmarks(self, results) -> List[List[Tuple[float, float, float]]]:
        """Devuelve landmarks normalizados (x, y, z) por cada mano detectada."""
        all_hands_landmarks: List[List[Tuple[float, float, float]]] = []

        if not results.multi_hand_landmarks:
            return all_hands_landmarks

        for hand_landmarks in results.multi_hand_landmarks:
            hand_points: List[Tuple[float, float, float]] = []
            for lm in hand_landmarks.landmark:
                hand_points.append((lm.x, lm.y, lm.z))
            all_hands_landmarks.append(hand_points)

        return all_hands_landmarks

    def close(self) -> None:
        """Libera recursos de MediaPipe."""
        self.hands.close()


def select_continuous_hand(
    landmarks: List[List[Tuple[float, float, float]]],
    previous_hand: np.ndarray | None = None,
) -> np.ndarray | None:
    """Select one hand while preserving continuity with the previous frame."""
    valid_hands: list[np.ndarray] = []
    for hand in landmarks:
        hand_array = np.asarray(hand, dtype=np.float32)
        if hand_array.shape == (LANDMARKS_PER_HAND, LANDMARK_DIMS):
            valid_hands.append(hand_array)

    if not valid_hands:
        return None
    if previous_hand is None or len(valid_hands) == 1:
        return valid_hands[0]

    previous = np.asarray(previous_hand, dtype=np.float32)
    if previous.shape != (LANDMARKS_PER_HAND, LANDMARK_DIMS):
        return valid_hands[0]

    return min(valid_hands, key=lambda hand: float(np.mean(np.linalg.norm(hand - previous, axis=1))))


def select_two_hand_slots(
    landmarks: List[List[Tuple[float, float, float]]],
    previous_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Return stable two-hand slots with empty hands filled as zeros."""
    valid_hands: list[np.ndarray] = []
    for hand in landmarks:
        hand_array = np.asarray(hand, dtype=np.float32)
        if hand_array.shape == (LANDMARKS_PER_HAND, LANDMARK_DIMS):
            valid_hands.append(hand_array)
    valid_hands = valid_hands[:HANDS_PER_FRAME]

    frame = np.zeros((HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS), dtype=np.float32)
    if not valid_hands:
        return frame

    previous = np.asarray(previous_frame, dtype=np.float32) if previous_frame is not None else None
    if previous is None or previous.shape != frame.shape or not np.any(np.abs(previous) > 1e-6):
        for slot, hand in enumerate(sorted(valid_hands, key=lambda item: float(np.mean(item[:, 0])))):
            frame[slot] = hand
        return frame

    assignments: list[tuple[float, tuple[int, ...]]] = []
    if len(valid_hands) == 1:
        for slot in range(HANDS_PER_FRAME):
            assignments.append((_slot_assignment_cost(previous, (valid_hands[0],), (slot,)), (slot,)))
    else:
        assignments.extend(
            [
                (_slot_assignment_cost(previous, tuple(valid_hands), (0, 1)), (0, 1)),
                (_slot_assignment_cost(previous, tuple(valid_hands), (1, 0)), (1, 0)),
            ]
        )

    _, best_slots = min(assignments, key=lambda item: item[0])
    for hand, slot in zip(valid_hands, best_slots):
        frame[slot] = hand
    return frame


def _slot_assignment_cost(previous: np.ndarray, hands: tuple[np.ndarray, ...], slots: tuple[int, ...]) -> float:
    cost = 0.0
    for hand, slot in zip(hands, slots):
        previous_hand = previous[slot]
        if np.any(np.abs(previous_hand) > 1e-6):
            cost += float(np.mean(np.linalg.norm(hand - previous_hand, axis=1)))
        else:
            cost += 0.25
    return cost
