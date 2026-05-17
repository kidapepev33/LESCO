"""Módulo para detección y seguimiento de manos con MediaPipe."""

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


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
            one_hand_points: List[Tuple[float, float, float]] = []
            for lm in hand_landmarks.landmark:
                one_hand_points.append((lm.x, lm.y, lm.z))
            all_hands_landmarks.append(one_hand_points)

        return all_hands_landmarks

    def close(self) -> None:
        """Libera recursos de MediaPipe."""
        self.hands.close()
