"""Runtime configuration for live continuous recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class LiveRecognitionConfig:
    """User-editable settings for the camera recognizer."""

    min_confidence: float = 0.70
    stride: int = 4
    no_hands_timeout_seconds: float = 0.9
    movement_threshold: float = 0.14
    pause_frames: int = 3
    min_clip_seconds: float = 0.1
    max_clip_seconds: float = 12.0
    use_prototypes: bool = True
    show_landmarks: bool = True
    save_debug_clips: bool = False
    save_clip_dir: str = "clips/debug"
    debug: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` when a setting is outside its safe range."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence debe estar entre 0.0 y 1.0.")
        if not 1 <= self.stride <= 30:
            raise ValueError("stride debe estar entre 1 y 30.")
        if not 0.1 <= self.no_hands_timeout_seconds <= 5.0:
            raise ValueError("no_hands_timeout_seconds debe estar entre 0.1 y 5.0.")
        if not 0.0 <= self.movement_threshold <= 1.0:
            raise ValueError("movement_threshold debe estar entre 0.0 y 1.0.")
        if not 1 <= self.pause_frames <= 30:
            raise ValueError("pause_frames debe estar entre 1 y 30.")
        if not 0.1 <= self.min_clip_seconds <= 10.0:
            raise ValueError("min_clip_seconds debe estar entre 0.1 y 10.0.")
        if not 1.0 <= self.max_clip_seconds <= 120.0:
            raise ValueError("max_clip_seconds debe estar entre 1.0 y 120.0.")
        if self.min_clip_seconds > self.max_clip_seconds:
            raise ValueError("min_clip_seconds no puede ser mayor que max_clip_seconds.")
        if not self.save_clip_dir.strip():
            raise ValueError("save_clip_dir no puede estar vacío.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveRecognitionConfig":
        """Build a config, ignoring unknown keys for forward compatibility."""
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config


def default_config_path(project_root: Path) -> Path:
    """Return the local JSON path used by the live recognizer."""
    return project_root / "config" / "live_runtime_config.json"


def load_runtime_config(path: Path) -> LiveRecognitionConfig:
    """Load config from JSON, or return defaults when it does not exist."""
    if not path.exists():
        config = LiveRecognitionConfig()
        config.validate()
        return config

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return LiveRecognitionConfig.from_dict(data)


def save_runtime_config(config: LiveRecognitionConfig, path: Path) -> None:
    """Validate and persist config as JSON."""
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
