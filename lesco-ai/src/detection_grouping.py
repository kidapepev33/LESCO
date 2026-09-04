"""Temporal grouping primitives for continuous sign detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from feature_extraction import SEQUENCE_LENGTH

EPSILON = 1e-8
SAME_WORD_IOU_THRESHOLD = 0.30
SAME_WORD_OVERLAP_RATIO_THRESHOLD = 0.50
SAME_WORD_CENTER_DISTANCE_FACTOR = 0.75
SAME_WORD_MAX_CHAIN_SPAN_FACTOR = 1.90
SAME_WORD_CHAIN_START_FACTOR = 0.85
StaticSegment = tuple[int, int, dict[str, object]]


@dataclass(frozen=True)
class WindowPrediction:
    """Raw prediction for one temporal window."""

    word: str
    confidence: float
    start_frame: int
    end_frame: int
    feature: np.ndarray
    static_signature: dict[str, object] | None = None


@dataclass(frozen=True)
class SignDetection:
    """Grouped detection after merging overlapping windows."""

    word: str
    confidence: float
    start_frame: int
    end_frame: int
    support: int
    feature: np.ndarray
    prototype_score: float | None = None
    static_signature: dict[str, object] | None = None
    static_score: float | None = None
    static_dominant_landmark: int | None = None
    static_dominance: float | None = None
    static_accepted: bool | None = None


def frame_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Temporal intersection-over-union for frame ranges."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def frame_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return overlap relative to the shorter temporal range."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0:
        return 0.0
    return intersection / shorter


def temporal_center(start_frame: int, end_frame: int) -> float:
    """Return the temporal center of a frame range."""
    return (start_frame + end_frame) / 2.0


def same_temporal_event(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
    max_gap_frames: int = 8,
) -> bool:
    """Decide whether two same-word detections explain the same gesture."""
    iou = frame_iou(a_start, a_end, b_start, b_end)
    overlap_ratio = frame_overlap_ratio(a_start, a_end, b_start, b_end)
    if iou >= SAME_WORD_IOU_THRESHOLD or overlap_ratio >= SAME_WORD_OVERLAP_RATIO_THRESHOLD:
        return True

    gap = max(a_start, b_start) - min(a_end, b_end)
    if gap > max_gap_frames:
        return False

    a_len = a_end - a_start
    b_len = b_end - b_start
    avg_len = (a_len + b_len) / 2.0
    center_distance = abs(temporal_center(a_start, a_end) - temporal_center(b_start, b_end))
    return center_distance <= avg_len * SAME_WORD_CENTER_DISTANCE_FACTOR


def sliding_window_ranges(
    num_frames: int,
    window_size: int = SEQUENCE_LENGTH,
    stride: int = 5,
) -> list[tuple[int, int]]:
    """Return overlapping temporal windows for a clip."""
    if num_frames < window_size:
        return [(0, num_frames)]

    ranges = [(start, start + window_size) for start in range(0, num_frames - window_size + 1, stride)]
    if ranges[-1][1] < num_frames:
        ranges.append((num_frames - window_size, num_frames))
    return ranges


def static_signature_for_range(
    start_frame: int,
    end_frame: int,
    static_segments: Sequence[StaticSegment] | None,
) -> dict[str, object] | None:
    """Return the static signature with strongest overlap for one temporal range."""
    if not static_segments:
        return None

    best_signature = None
    best_overlap = 0.0
    for segment_start, segment_end, signature in static_segments:
        overlap = frame_overlap_ratio(start_frame, end_frame, segment_start, segment_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_signature = signature
    return best_signature if best_overlap > 0.0 else None


def group_repeated_detections(
    predictions: Sequence[WindowPrediction],
    max_gap_frames: int = 8,
) -> list[SignDetection]:
    """Merge repeated detections caused by overlapping windows."""
    by_word: dict[str, list[WindowPrediction]] = {}
    for pred in sorted(predictions, key=lambda p: (p.word, p.start_frame, -p.confidence)):
        by_word.setdefault(pred.word, []).append(pred)

    grouped: list[SignDetection] = []
    for word, items in by_word.items():
        active: list[WindowPrediction] = []
        for pred in items:
            if not active:
                active = [pred]
                continue

            current_start = min(p.start_frame for p in active)
            current_end = max(p.end_frame for p in active)

            same_event = same_temporal_event(current_start, current_end, pred.start_frame, pred.end_frame, max_gap_frames)
            too_long_chain = _would_create_overlong_window_chain(active, pred)

            if same_event and not too_long_chain:
                active.append(pred)
            else:
                grouped.append(_merge_group(word, active))
                active = [pred]

        if active:
            grouped.append(_merge_group(word, active))

    return sorted(grouped, key=lambda d: (d.start_frame, d.end_frame, -d.confidence))


def _would_create_overlong_window_chain(active: Sequence[WindowPrediction], pred: WindowPrediction) -> bool:
    """Return true when overlapping windows likely cover a second execution."""
    current_start = min(p.start_frame for p in active)
    current_end = max(p.end_frame for p in active)
    combined_span = max(current_end, pred.end_frame) - min(current_start, pred.start_frame)
    durations = [p.end_frame - p.start_frame for p in active]
    durations.append(pred.end_frame - pred.start_frame)
    avg_duration = max(float(np.mean(durations)), EPSILON)
    start_separation = pred.start_frame - current_start
    return (
        combined_span > avg_duration * SAME_WORD_MAX_CHAIN_SPAN_FACTOR
        and start_separation >= avg_duration * SAME_WORD_CHAIN_START_FACTOR
    )


def _merge_group(word: str, group: Sequence[WindowPrediction]) -> SignDetection:
    weights = np.asarray([p.confidence for p in group], dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), EPSILON)
    feature = np.sum(np.stack([p.feature for p in group]) * weights[:, None, None], axis=0).astype(np.float32)
    static_signature = max(group, key=lambda p: p.confidence).static_signature
    return SignDetection(
        word=word,
        confidence=max(p.confidence for p in group),
        start_frame=min(p.start_frame for p in group),
        end_frame=max(p.end_frame for p in group),
        support=len(group),
        feature=feature,
        static_signature=static_signature,
        static_dominant_landmark=None if static_signature is None else int(static_signature["dominant_landmark"]),
        static_dominance=None if static_signature is None else float(static_signature["dominance"]),
        static_accepted=None if static_signature is None else bool(static_signature["accepted"]),
    )
