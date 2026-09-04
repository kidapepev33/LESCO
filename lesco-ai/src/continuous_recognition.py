"""Continuous sign recognition engine for LESCO sentence clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tensorflow import keras

from dataset_utils import get_default_dataset_dir
from detection_grouping import (
    EPSILON,
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
from feature_extraction import SEQUENCE_LENGTH, extract_landmark_features, static_landmark_signature, validate_raw_sequence
from model_utils import load_label_map, load_sign_model

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
    stride: int = 5,
    min_confidence: float = 0.55,
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


class PrototypeLibrary:
    """Class prototypes used for final visual validation."""

    def __init__(
        self,
        prototypes: dict[str, np.ndarray],
        radii: dict[str, float],
        static_prototypes: dict[str, np.ndarray] | None = None,
        static_radii: dict[str, float] | None = None,
    ) -> None:
        self.prototypes = prototypes
        self.radii = radii
        self.static_prototypes = static_prototypes if static_prototypes is not None else {}
        self.static_radii = static_radii if static_radii is not None else {}

    @classmethod
    def from_dataset(cls, dataset_dir: Path | None = None) -> "PrototypeLibrary":
        if dataset_dir is None:
            dataset_dir = get_default_dataset_dir()

        prototypes: dict[str, np.ndarray] = {}
        radii: dict[str, float] = {}
        static_prototypes: dict[str, np.ndarray] = {}
        static_radii: dict[str, float] = {}
        for label_dir in sorted(Path(dataset_dir).iterdir()):
            if not label_dir.is_dir():
                continue

            features = []
            static_features = []
            for sample_file in sorted(label_dir.glob("sample_*.npy")):
                sample = np.load(sample_file)
                features.append(extract_landmark_features(sample))
                static_window = sample[-min(5, len(sample)) :]
                signature = static_landmark_signature(static_window)
                if bool(signature["accepted"]):
                    static_features.append(np.asarray(signature["vector"], dtype=np.float32))
            if not features:
                continue

            stacked = np.asarray(features, dtype=np.float32)
            prototype = np.mean(stacked, axis=0).astype(np.float32)
            distances = np.linalg.norm((stacked - prototype).reshape(len(stacked), -1), axis=1)
            prototypes[label_dir.name] = prototype
            radii[label_dir.name] = max(float(np.percentile(distances, 75)), EPSILON)

            if static_features:
                static_stacked = np.asarray(static_features, dtype=np.float32)
                static_prototype = np.mean(static_stacked, axis=0).astype(np.float32)
                static_distances = np.linalg.norm(static_stacked - static_prototype, axis=1)
                static_prototypes[label_dir.name] = static_prototype
                static_radii[label_dir.name] = max(float(np.percentile(static_distances, 75)), EPSILON)

        return cls(
            prototypes=prototypes,
            radii=radii,
            static_prototypes=static_prototypes,
            static_radii=static_radii,
        )

    def score(self, word: str, feature: np.ndarray) -> float:
        """Return a bounded visual compatibility score in ``(0, 1]``."""
        prototype = self.prototypes.get(word)
        if prototype is None:
            return 0.0
        radius = self.radii[word]
        distance = float(np.linalg.norm((feature - prototype).reshape(-1)))
        return float(np.exp(-distance / (radius + EPSILON)))

    def static_score(self, word: str, signature: dict[str, object] | None) -> float | None:
        """Return static landmark compatibility, or ``None`` when unavailable."""
        if signature is None or not bool(signature.get("accepted", False)):
            return None
        prototype = self.static_prototypes.get(word)
        if prototype is None:
            return None
        radius = self.static_radii[word]
        vector = np.asarray(signature["vector"], dtype=np.float32)
        distance = float(np.linalg.norm(vector - prototype))
        return float(np.exp(-distance / (radius + EPSILON)))


class SentenceBuilder:
    """Small local beam-search sentence builder."""

    def __init__(
        self,
        beam_width: int = 5,
        visual_weight: float = 4.0,
        language_weight: float = 1.0,
        skip_penalty: float = 0.25,
    ) -> None:
        self.beam_width = beam_width
        self.visual_weight = visual_weight
        self.language_weight = language_weight
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

            for total, visual_total, language_total, chosen in beams:
                if chosen and det.start_frame <= chosen[-1].start_frame + 3:
                    continue

                transition = self._transition_score(chosen[-1].word if chosen else None, det.word)
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
                new_language = language_total + transition + repeat_penalty + overlap_penalty
                new_total = (
                    total
                    + self.visual_weight * visual
                    + self.language_weight * transition
                    + repeat_penalty
                    + overlap_penalty
                )
                next_beams.append((new_total, new_visual, new_language, chosen + [det]))

            beams = sorted(next_beams, key=lambda item: item[0] - self.skip_penalty * (len(scored) - len(item[3])), reverse=True)[
                : self.beam_width
            ]

        best_total, best_visual, best_language, best_detections = max(beams, key=lambda item: item[0])
        best_detections = self._remove_transition_repeats(best_detections)
        best_detections = self._resolve_consecutive_duplicates(best_detections, all_scored)
        best_detections = self._prefer_common_phrase_order(best_detections)
        words = tuple(det.word for det in best_detections)
        return SentenceResult(
            sentence=" ".join(words).upper(),
            words=words,
            detections=tuple(best_detections),
            candidates=tuple(scored),
            visual_score=best_visual,
            language_score=best_language,
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

    def _transition_score(self, previous: str | None, current: str) -> float:
        if previous is None:
            return 0.0
        if previous == current:
            return -0.35
        common_pairs = {
            ("yo", "tener"): 0.45,
            ("tener", "bano"): 0.35,
            ("yo", "querer"): 0.35,
            ("querer", "agua"): 0.35,
            ("hola", "usted"): 0.20,
            ("hola", "yo"): 0.15,
        }
        return common_pairs.get((previous, current), 0.0)

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
                    and frame_iou(det.start_frame, det.end_frame, next_det.start_frame, next_det.end_frame) >= 0.30
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

    def _prefer_common_phrase_order(self, detections: Sequence[SignDetection]) -> list[SignDetection]:
        """Prefer the common YO TENER order when both detections are adjacent."""
        ordered = list(detections)
        index = 0
        while index < len(ordered) - 1:
            current = ordered[index]
            following = ordered[index + 1]
            if (
                current.word == "tener"
                and following.word == "yo"
                and frame_overlap_ratio(
                    current.start_frame,
                    current.end_frame,
                    following.start_frame,
                    following.end_frame,
                )
                >= 0.40
            ):
                ordered[index], ordered[index + 1] = following, current
                index += 2
                continue
            index += 1
        return ordered


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
        self.model = model if model is not None else load_sign_model()
        self.index_to_label = index_to_label if index_to_label is not None else load_label_map()
        self.prototypes = prototypes
        self.builder = builder if builder is not None else SentenceBuilder()

    def recognize(
        self,
        sequence: np.ndarray,
        window_size: int = SEQUENCE_LENGTH,
        stride: int = 5,
        min_confidence: float = 0.55,
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
            transition = start[None, :, :] * (1.0 - alpha[:, None, None]) + end[None, :, :] * alpha[:, None, None]
            pieces.append(transition.astype(np.float32))
        pieces.append(sample.astype(np.float32))
    return np.concatenate(pieces, axis=0)
