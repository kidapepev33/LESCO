"""Continuous sign recognition engine for LESCO sentence clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from src.config.runtime import MIN_CONFIDENCE, WINDOW_STRIDE
from src.recognition.grouping import (
    SAME_WORD_CENTER_DISTANCE_FACTOR,
    SAME_WORD_CHAIN_START_FACTOR,
    SAME_WORD_IOU_THRESHOLD,
    SAME_WORD_MAX_CHAIN_SPAN_FACTOR,
    SAME_WORD_OVERLAP_RATIO_THRESHOLD,
    SignDetection,
    StaticSegment,
    WindowPrediction,
    _merge_group,
    _would_create_overlong_window_chain,
    frame_iou,
    frame_overlap_ratio,
    group_repeated_detections,
    same_temporal_event,
    sliding_window_ranges,
    static_signature_for_range,
    temporal_center,
)
from src.recognition.prototypes import PrototypeLibrary
from src.vision.features import SEQUENCE_LENGTH, extract_landmark_features, validate_raw_sequence

if TYPE_CHECKING:
    from tensorflow import keras


@dataclass(frozen=True)
class SentenceResult:
    """Final continuous-recognition output."""

    sentence: str
    words: tuple[str, ...]
    detections: tuple[SignDetection, ...]
    candidates: tuple[SignDetection, ...]
    visual_score: float
    language_score: float
    total_score: float


def extract_window_features(
    sequence: np.ndarray,
    window_ranges: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Extract model features for all windows once."""
    sequence = validate_raw_sequence(sequence)
    features = [extract_landmark_features(sequence[start:end]) for start, end in window_ranges]
    return np.asarray(features, dtype=np.float32)


def predict_windows(
    sequence: np.ndarray,
    model: keras.Model,
    index_to_label: dict[int, str],
    window_size: int = SEQUENCE_LENGTH,
    stride: int = WINDOW_STRIDE,
    min_confidence: float = MIN_CONFIDENCE,
    top_k: int = 2,
    static_segments: Sequence[StaticSegment] | None = None,
) -> list[WindowPrediction]:
    """Predict candidate signs on overlapping windows."""
    ranges = sliding_window_ranges(len(sequence), window_size=window_size, stride=stride)
    features = extract_window_features(sequence, ranges)
    probs_batch = model.predict(features, verbose=0)

    predictions: list[WindowPrediction] = []
    for (start, end), feature, probs in zip(ranges, features, probs_batch):
        static_signature = static_signature_for_range(start, end, static_segments)
        top_indices = np.argsort(probs)[::-1][:top_k]
        for idx in top_indices:
            confidence = float(probs[idx])
            if confidence < min_confidence:
                continue
            predictions.append(
                WindowPrediction(
                    word=index_to_label.get(int(idx), f"Clase {int(idx)}"),
                    confidence=confidence,
                    start_frame=start,
                    end_frame=end,
                    feature=feature,
                    static_signature=static_signature,
                )
            )
    return predictions


class SentenceBuilder:
    """Build a sentence using visual evidence and label-agnostic timing rules."""

    def __init__(
        self,
        beam_width: int = 5,
        visual_weight: float = 4.0,
        skip_penalty: float = 0.25,
    ) -> None:
        self.beam_width = beam_width
        self.visual_weight = visual_weight
        self.skip_penalty = skip_penalty

    def build(
        self,
        detections: Sequence[SignDetection],
        prototypes: PrototypeLibrary | None = None,
        suppress_competing: bool = True,
    ) -> SentenceResult:
        """Choose the most plausible sentence from temporal detections."""
        if not detections:
            return SentenceResult("", (), (), (), 0.0, 0.0, 0.0)

        all_scored = [
            replace(
                det,
                prototype_score=prototypes.score(det.word, det.feature) if prototypes else None,
                static_score=prototypes.static_score(det.word, det.static_signature) if prototypes else None,
            )
            for det in detections
        ]
        scored = suppress_competing_detections(all_scored) if suppress_competing else list(all_scored)

        beams: list[tuple[float, float, float, list[SignDetection]]] = [(0.0, 0.0, 0.0, [])]
        for det in scored:
            next_beams = beams[:]
            visual = self._visual_score(det)

            for total, visual_total, temporal_total, chosen in beams:
                if chosen and det.start_frame <= chosen[-1].start_frame + 3:
                    continue

                overlap_penalty = 0.0
                if chosen:
                    overlap_penalty = -0.55 * frame_iou(
                        chosen[-1].start_frame,
                        chosen[-1].end_frame,
                        det.start_frame,
                        det.end_frame,
                    )
                repeat_ok = self._repeat_is_real(chosen[-1], det) if chosen else True
                repeat_penalty = 0.0 if repeat_ok else -1.2
                new_visual = visual_total + visual
                new_temporal = temporal_total + repeat_penalty + overlap_penalty
                new_total = total + self.visual_weight * visual + repeat_penalty + overlap_penalty
                next_beams.append((new_total, new_visual, new_temporal, chosen + [det]))

            beams = sorted(next_beams, key=lambda item: item[0] - self.skip_penalty * (len(scored) - len(item[3])), reverse=True)[
                : self.beam_width
            ]

        best_total, best_visual, best_temporal, best_detections = max(beams, key=lambda item: item[0])
        best_detections = self._remove_transition_repeats(best_detections)
        best_detections = self._resolve_consecutive_duplicates(best_detections, all_scored)
        words = tuple(det.word for det in best_detections)
        return SentenceResult(
            sentence=" ".join(words).upper(),
            words=words,
            detections=tuple(best_detections),
            candidates=tuple(scored),
            visual_score=best_visual,
            # Kept under the legacy field name for compatibility. This now
            # contains only label-agnostic temporal penalties.
            language_score=best_temporal,
            total_score=best_total,
        )

    def _visual_score(self, detection: SignDetection) -> float:
        confidence_score = detection.confidence
        prototype_score = 0.0
        if detection.prototype_score is not None:
            prototype_score = detection.prototype_score
        movement_score = confidence_score + 0.35 * prototype_score + min(detection.support, 4) * 0.05
        if detection.static_score is None:
            return float(movement_score)
        return float(0.8 * movement_score + 0.2 * detection.static_score)

    def _repeat_is_real(self, previous: SignDetection, current: SignDetection) -> bool:
        gap = current.start_frame - previous.end_frame
        min_separation = max(4, SEQUENCE_LENGTH // 3)
        return previous.word != current.word or gap >= min_separation

    def _remove_transition_repeats(self, detections: Sequence[SignDetection]) -> list[SignDetection]:
        """Remove repeated labels that are better explained as a transition."""
        cleaned: list[SignDetection] = []
        for index, det in enumerate(detections):
            if cleaned and cleaned[-1].word == det.word:
                previous = cleaned[-1]
                next_det = detections[index + 1] if index + 1 < len(detections) else None
                overlaps_previous = frame_iou(
                    previous.start_frame,
                    previous.end_frame,
                    det.start_frame,
                    det.end_frame,
                )
                overlaps_next = (
                    next_det is not None
                    and next_det.word != det.word
                    and frame_overlap_ratio(
                        det.start_frame,
                        det.end_frame,
                        next_det.start_frame,
                        next_det.end_frame,
                    )
                    >= 0.30
                )
                if overlaps_previous >= 0.25 and overlaps_next:
                    continue
            cleaned.append(det)
        return cleaned

    def _resolve_consecutive_duplicates(
        self,
        detections: Sequence[SignDetection],
        candidates: Sequence[SignDetection],
    ) -> list[SignDetection]:
        """Replace stuck adjacent repeated words with a nearby alternative when available."""
        cleaned: list[SignDetection] = []
        for det in detections:
            if cleaned and cleaned[-1].word == det.word:
                previous = cleaned[-1]
                gap = det.start_frame - previous.end_frame
                if 0 <= gap <= max(4, SEQUENCE_LENGTH // 3):
                    alternative = self._best_duplicate_alternative(previous, det, candidates)
                    if alternative is not None:
                        cleaned.append(alternative)
                    elif (det.confidence, det.support) > (previous.confidence, previous.support):
                        cleaned[-1] = det
                    continue
            cleaned.append(det)
        return cleaned

    def _best_duplicate_alternative(
        self,
        previous: SignDetection,
        duplicate: SignDetection,
        candidates: Sequence[SignDetection],
    ) -> SignDetection | None:
        alternatives = [
            candidate
            for candidate in candidates
            if candidate.word != duplicate.word
            and candidate.word != previous.word
            and frame_overlap_ratio(
                duplicate.start_frame,
                duplicate.end_frame,
                candidate.start_frame,
                candidate.end_frame,
            )
            >= 0.50
        ]
        if not alternatives:
            return None
        return max(alternatives, key=detection_score)

def detection_score(detection: SignDetection) -> float:
    """Score used to compare candidates that explain the same frames."""
    prototype_score = detection.prototype_score if detection.prototype_score is not None else 0.0
    movement_score = detection.confidence + 0.45 * prototype_score + min(detection.support, 4) * 0.03
    if detection.static_score is None:
        return movement_score
    return 0.8 * movement_score + 0.2 * detection.static_score


def suppress_competing_detections(
    detections: Sequence[SignDetection],
    cross_label_iou: float = 0.45,
) -> list[SignDetection]:
    """Drop lower-scoring candidates that cover the same temporal evidence."""
    kept: list[SignDetection] = []
    filtered = [det for det in detections if det.support > 1 or det.confidence >= 0.80]
    for det in sorted(filtered, key=detection_score, reverse=True):
        should_drop = False
        for chosen in kept:
            if det.word == chosen.word:
                continue
            if frame_iou(det.start_frame, det.end_frame, chosen.start_frame, chosen.end_frame) >= cross_label_iou:
                should_drop = True
                break
        if not should_drop:
            kept.append(det)
    return sorted(kept, key=lambda item: (item.start_frame, item.end_frame, -item.confidence))


class ContinuousRecognizer:
    """End-to-end recognizer for clips containing multiple signs."""

    def __init__(
        self,
        model: keras.Model | None = None,
        index_to_label: dict[int, str] | None = None,
        prototypes: PrototypeLibrary | None = None,
        builder: SentenceBuilder | None = None,
    ) -> None:
        if model is None:
            from src.recognition.model import load_sign_model

            model = load_sign_model()
        if index_to_label is None:
            from src.recognition.model import load_label_map

            index_to_label = load_label_map()
        self.model = model
        self.index_to_label = index_to_label
        self.prototypes = prototypes
        self.builder = builder if builder is not None else SentenceBuilder()

    def recognize(
        self,
        sequence: np.ndarray,
        window_size: int = SEQUENCE_LENGTH,
        stride: int = WINDOW_STRIDE,
        min_confidence: float = MIN_CONFIDENCE,
        top_k: int = 2,
        static_segments: Sequence[StaticSegment] | None = None,
    ) -> SentenceResult:
        raw_predictions = predict_windows(
            sequence,
            model=self.model,
            index_to_label=self.index_to_label,
            window_size=window_size,
            stride=stride,
            min_confidence=min_confidence,
            top_k=top_k,
            static_segments=static_segments,
        )
        detections = group_repeated_detections(raw_predictions)
        return self.builder.build(detections, prototypes=self.prototypes)


def concatenate_samples(samples: Iterable[np.ndarray], transition_frames: int = 0) -> np.ndarray:
    """Concatenate isolated signs into one synthetic clip."""
    pieces: list[np.ndarray] = []
    for sample in samples:
        sample = validate_raw_sequence(sample)
        if pieces and transition_frames > 0:
            start = pieces[-1][-1]
            end = sample[0]
            alpha = np.linspace(0.0, 1.0, num=transition_frames + 2, dtype=np.float32)[1:-1]
            weights = alpha.reshape((transition_frames,) + (1,) * start.ndim)
            transition = start[None, ...] * (1.0 - weights) + end[None, ...] * weights
            pieces.append(transition.astype(np.float32))
        pieces.append(sample.astype(np.float32))
    return np.concatenate(pieces, axis=0)
