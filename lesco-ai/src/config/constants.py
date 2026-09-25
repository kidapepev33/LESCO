"""Configuraciones base del proyecto LESCO-AI."""

# Formato estructural de landmarks y entrada temporal del modelo.
SEQUENCE_LENGTH = 30
HANDS_PER_FRAME = 2
LANDMARKS_PER_HAND = 21
LANDMARK_DIMS = 3
ONE_HAND_FRAME_SHAPE = (LANDMARKS_PER_HAND, LANDMARK_DIMS)
TWO_HAND_FRAME_SHAPE = (HANDS_PER_FRAME, LANDMARKS_PER_HAND, LANDMARK_DIMS)

# Nombres de los artefactos compatibles con el modelo actual.
MODEL_FILENAME = "lesco_landmark_lstm.keras"
LABEL_MAP_FILENAME = "label_map.json"

# Índice de cámara (0 suele ser la cámara por defecto)
CAMERA_INDEX = 0

# Resolución deseada para captura (opcional)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configuración de MediaPipe Hands
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
