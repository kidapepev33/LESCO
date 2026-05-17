"""Configuraciones base del proyecto LESCO-AI."""

# Índice de cámara (0 suele ser la cámara por defecto)
CAMERA_INDEX = 0

# Resolución deseada para captura (opcional)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configuración de MediaPipe Hands
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
