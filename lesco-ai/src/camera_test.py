"""Prueba básica de cámara y detección de manos para LESCO-AI."""

import cv2

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from hand_tracker import HandTracker


def main() -> None:
    """Abre la cámara, detecta manos y dibuja landmarks en tiempo real."""
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        print("Verifica que la cámara esté conectada y que el índice sea correcto en src/config.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    print("Iniciando cámara. Presiona 'q' para salir.")
    hand_was_detected = False

    try:
        while True:
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not success:
                print("[WARNING] No se pudo leer un frame de la cámara.")
                break

            results = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, results)

            normalized_landmarks = tracker.get_normalized_landmarks(results)
            hand_is_detected = len(normalized_landmarks) > 0
            if hand_is_detected and not hand_was_detected:
                print(f"Mano detectada. Cantidad: {len(normalized_landmarks)}")
            elif not hand_is_detected and hand_was_detected:
                print("No se detectan manos en este momento.")
            hand_was_detected = hand_is_detected

            cv2.imshow("LESCO-AI | Hand Tracking Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Programa finalizado.")


if __name__ == "__main__":
    main()
