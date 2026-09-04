"""Live segment classification and buffering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import sys

import numpy as np

from continuous_recognition import ContinuousRecognizer, SentenceBuilder, SentenceResult, SignDetection
from feature_extraction import extract_landmark_features
from runtime_config import LiveRecognitionConfig


@dataclass
class SegmentPrediction:
    """Classification result for one provisional segment."""

    sequence: np.ndarray
    result: SentenceResult
    confidence: float
    model_confidence: float
    static_signature: dict[str, object] | None = None
    static_score: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    raw_top_predictions: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RawSegmentDebugEntry:
    """Raw model output for one classified live segment."""

    segment_number: int
    frame_count: int
    top_predictions: tuple[tuple[str, float], ...]
    accepted: bool
    threshold: float
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass
class SegmentDecision:
    """Decision for one independently classified segment."""

    accepted: SegmentPrediction | None = None
    individual_confidence: float | None = None


@dataclass
class SegmentPredictionBuffer:
    """Collect accepted independent segment predictions for the current sentence."""

    config: LiveRecognitionConfig
    accepted_segments: list[SegmentPrediction] = field(default_factory=list)
    raw_debug_entries: list[RawSegmentDebugEntry] = field(default_factory=list)
    detected_segments: int = 0
    last_individual_confidence: float | None = None
    last_static_signature: dict[str, object] | None = None
    last_static_score: float | None = None

    def submit(
        self,
        sequence: np.ndarray,
        recognizer: ContinuousRecognizer,
        static_signature: dict[str, object] | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        movement_exit: bool = False,
    ) -> SegmentDecision:
        self.detected_segments += 1
        individual = _classify_segment_compat(sequence, recognizer, static_signature=static_signature)
        individual = replace(individual, start_frame=start_frame, end_frame=end_frame)

        accepted: SegmentPrediction | None = None
        if individual.confidence >= self.config.min_confidence:
            accepted = individual

        self.raw_debug_entries.append(
            RawSegmentDebugEntry(
                segment_number=self.detected_segments,
                frame_count=len(sequence),
                top_predictions=individual.raw_top_predictions,
                accepted=accepted is not None,
                threshold=self.config.min_confidence,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        self.raw_debug_entries = self.raw_debug_entries[-5:]

        if accepted is not None:
            self.accepted_segments.append(accepted)

        self.last_individual_confidence = individual.confidence
        self.last_static_signature = static_signature
        self.last_static_score = individual.static_score

        return SegmentDecision(
            accepted=accepted,
            individual_confidence=self.last_individual_confidence,
        )

    def reset_sentence(self) -> None:
        self.accepted_segments.clear()
        self.detected_segments = 0
        self.last_individual_confidence = None
        self.last_static_signature = None
        self.last_static_score = None

    def reset_debug_sentence(self) -> None:
        self.reset_sentence()
        self.raw_debug_entries.clear()

    def build_sentence_result(self) -> SentenceResult:
        if not self.accepted_segments:
            return SentenceResult("", (), (), (), 0.0, 0.0, 0.0)

        detections: list[SignDetection] = []
        visual_scores: list[float] = []
        frame_offset = 0
        for segment in self.accepted_segments:
            visual_scores.append(segment.confidence)
            segment_offset = segment.start_frame if segment.start_frame is not None else frame_offset
            for detection in segment.result.detections:
                detections.append(
                    SignDetection(
                        word=detection.word,
                        confidence=detection.confidence,
                        start_frame=segment_offset + detection.start_frame,
                        end_frame=segment_offset + detection.end_frame,
                        support=detection.support,
                        feature=detection.feature,
                        prototype_score=detection.prototype_score,
                        static_signature=detection.static_signature,
                        static_score=detection.static_score,
                        static_dominant_landmark=detection.static_dominant_landmark,
                        static_dominance=detection.static_dominance,
                        static_accepted=detection.static_accepted,
                    )
                )
            frame_offset += len(segment.sequence)

        result = SentenceBuilder().build(detections, suppress_competing=False)
        visual_score = float(np.mean(visual_scores)) if visual_scores else 0.0
        return SentenceResult(
            sentence=result.sentence,
            words=result.words,
            detections=result.detections,
            candidates=result.candidates,
            visual_score=visual_score,
            language_score=result.language_score,
            total_score=visual_score + result.language_score,
        )


def classify_segment(
    sequence: np.ndarray,
    recognizer: ContinuousRecognizer,
    static_signature: dict[str, object] | None = None,
) -> SegmentPrediction:
    """Classify one provisional segment with the existing model feature pipeline."""
    sequence = np.asarray(sequence, dtype=np.float32)
    feature = extract_landmark_features(sequence)
    probs = recognizer.model.predict(feature[None, ...], verbose=0)[0]
    raw_top_predictions = top_model_predictions(probs, recognizer.index_to_label, top_k=3)
    index = int(np.argmax(probs))
    model_confidence = float(probs[index])
    word = recognizer.index_to_label.get(index, f"Clase {index}")
    static_score = recognizer.prototypes.static_score(word, static_signature) if recognizer.prototypes else None
    confidence = model_confidence if static_score is None else float(0.5 * model_confidence + 0.5 * static_score)
    detection = SignDetection(
        word=word,
        confidence=confidence,
        start_frame=0,
        end_frame=len(sequence),
        support=1,
        feature=feature,
        prototype_score=recognizer.prototypes.score(word, feature) if recognizer.prototypes else None,
        static_score=static_score,
        static_dominant_landmark=None if static_signature is None else int(static_signature["dominant_landmark"]),
        static_dominance=None if static_signature is None else float(static_signature["dominance"]),
        static_accepted=None if static_signature is None else bool(static_signature["accepted"]),
    )
    result = SentenceResult(
        sentence=word.upper(),
        words=(word,),
        detections=(detection,),
        candidates=(detection,),
        visual_score=confidence,
        language_score=0.0,
        total_score=confidence,
    )
    return SegmentPrediction(
        sequence=sequence,
        result=result,
        confidence=confidence,
        model_confidence=model_confidence,
        static_signature=static_signature,
        static_score=static_score,
        raw_top_predictions=raw_top_predictions,
    )


def _classify_segment_compat(
    sequence: np.ndarray,
    recognizer: ContinuousRecognizer,
    static_signature: dict[str, object] | None = None,
) -> SegmentPrediction:
    predict_live = sys.modules.get("predict_live")
    patched = getattr(predict_live, "classify_segment", classify_segment) if predict_live is not None else classify_segment
    return patched(sequence, recognizer, static_signature=static_signature)


def top_model_predictions(
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    top_k: int = 3,
) -> tuple[tuple[str, float], ...]:
    """Return raw model top-k labels before runtime filters."""
    probs = np.asarray(probabilities, dtype=np.float32)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return tuple((index_to_label.get(int(index), f"Clase {int(index)}"), float(probs[index])) for index in top_indices)
