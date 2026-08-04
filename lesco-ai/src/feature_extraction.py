"""Feature extraction for robust LESCO sign recognition.

The model must not learn where the hand is in the camera frame. This module
converts raw MediaPipe landmarks into hand-centered, scale-normalized features
and is used by both training and live prediction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

SEQUENCE_LENGTH = 30
LANDMARKS_PER_FRAME = 21
COORDS_PER_LANDMARK = 3
RAW_FRAME_SHAPE = (LANDMARKS_PER_FRAME, COORDS_PER_LANDMARK)
RAW_FEATURE_SIZE = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK

WRIST = 0
PALM_REFERENCE_LANDMARKS = (5, 9, 13, 17)
EPSILON = 1e-6

HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

NORMALIZED_COORD_FEATURES = RAW_FEATURE_SIZE
BONE_DISTANCE_FEATURES = len(HAND_BONES)
LOCAL_VELOCITY_FEATURES = RAW_FEATURE_SIZE
LOCAL_ACCELERATION_FEATURES = RAW_FEATURE_SIZE
WRIST_TRAJECTORY_FEATURES = COORDS_PER_LANDMARK
WRIST_VELOCITY_FEATURES = COORDS_PER_LANDMARK
WRIST_ACCELERATION_FEATURES = COORDS_PER_LANDMARK
FEATURE_SIZE = (
    NORMALIZED_COORD_FEATURES
    + BONE_DISTANCE_FEATURES
    + LOCAL_VELOCITY_FEATURES
    + LOCAL_ACCELERATION_FEATURES
    + WRIST_TRAJECTORY_FEATURES
    + WRIST_VELOCITY_FEATURES
    + WRIST_ACCELERATION_FEATURES
)

FEATURE_PIPELINE_NAME = "relative_hand_shape_with_start_relative_wrist_trajectory_v2"


def validate_raw_sequence(sequence: np.ndarray) -> np.ndarray:
    """Return a float32 landmark sequence after validating its shape."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != RAW_FRAME_SHAPE:
        raise ValueError(
            "Shape de secuencia inválido. "
            f"Esperado: (frames, {LANDMARKS_PER_FRAME}, {COORDS_PER_LANDMARK}), "
            f"obtenido: {sequence.shape}"
        )
    if sequence.shape[0] < 1:
        raise ValueError("La secuencia no tiene frames para procesar.")
    return sequence


def temporal_resample(sequence: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Resample a raw landmark sequence to a fixed temporal length."""
    sequence = validate_raw_sequence(sequence)
    frames = sequence.shape[0]

    if frames == target_len:
        return sequence.astype(np.float32)

    sequence_flat = sequence.reshape(frames, RAW_FEATURE_SIZE)
    old_t = np.linspace(0.0, 1.0, num=frames, dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    resampled_flat = np.stack(
        [np.interp(new_t, old_t, sequence_flat[:, i]) for i in range(RAW_FEATURE_SIZE)],
        axis=1,
    ).astype(np.float32)

    return resampled_flat.reshape(target_len, LANDMARKS_PER_FRAME, COORDS_PER_LANDMARK)


def palm_scale(frame: np.ndarray) -> float:
    """Estimate hand scale from wrist-to-MCP distances."""
    wrist = frame[WRIST]
    reference_points = frame[list(PALM_REFERENCE_LANDMARKS)]
    distances = np.linalg.norm(reference_points - wrist, axis=1)
    scale = float(np.mean(distances))
    return max(scale, EPSILON)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Center a frame on the wrist and divide it by the hand scale."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != RAW_FRAME_SHAPE:
        raise ValueError(f"Frame inválido. Esperado {RAW_FRAME_SHAPE}, obtenido {frame.shape}")

    wrist = frame[WRIST]
    relative = frame - wrist
    return (relative / palm_scale(frame)).astype(np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Apply wrist-relative, palm-scale normalization to every frame."""
    sequence = validate_raw_sequence(sequence)
    return np.stack([normalize_frame(frame) for frame in sequence], axis=0).astype(np.float32)


def reference_scale(sequence: np.ndarray) -> float:
    """Compute a stable sequence scale from all available palm scales."""
    sequence = validate_raw_sequence(sequence)
    scales = [palm_scale(frame) for frame in sequence]
    return max(float(np.mean(scales)), EPSILON)


def bone_distances(normalized_sequence: np.ndarray) -> np.ndarray:
    """Compute normalized distances along MediaPipe hand bones."""
    distances = []
    for start, end in HAND_BONES:
        distances.append(np.linalg.norm(normalized_sequence[:, end] - normalized_sequence[:, start], axis=1))
    return np.stack(distances, axis=1).astype(np.float32)


def temporal_deltas(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute first and second temporal differences with stable sequence length."""
    velocity = np.diff(values, axis=0, prepend=values[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    return velocity.astype(np.float32), acceleration.astype(np.float32)


def wrist_trajectory(sequence: np.ndarray) -> np.ndarray:
    """Return wrist movement relative to the first frame, normalized by hand size."""
    sequence = validate_raw_sequence(sequence)
    wrists = sequence[:, WRIST, :]
    trajectory = (wrists - wrists[:1]) / reference_scale(sequence)
    return trajectory.astype(np.float32)


def extract_landmark_features(sequence: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Convert raw landmarks into model-ready invariant features.

    Input: ``(frames, 21, 3)`` raw MediaPipe landmarks.
    Output: ``(target_len, FEATURE_SIZE)``.
    """
    resampled = temporal_resample(sequence, target_len=target_len)
    normalized = normalize_sequence(resampled)

    coords = normalized.reshape(target_len, RAW_FEATURE_SIZE)
    bones = bone_distances(normalized)
    local_velocity, local_acceleration = temporal_deltas(coords)
    trajectory = wrist_trajectory(resampled)
    wrist_velocity, wrist_acceleration = temporal_deltas(trajectory)

    return np.concatenate(
        [
            coords,
            bones,
            local_velocity,
            local_acceleration,
            trajectory,
            wrist_velocity,
            wrist_acceleration,
        ],
        axis=1,
    ).astype(np.float32)


def transform_sequence(
    sequence: np.ndarray,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    """Apply a synthetic camera-position transform to raw landmarks."""
    sequence = validate_raw_sequence(sequence)
    offset_array = np.asarray(offset, dtype=np.float32).reshape(1, 1, COORDS_PER_LANDMARK)
    return (sequence * np.float32(scale) + offset_array).astype(np.float32)
