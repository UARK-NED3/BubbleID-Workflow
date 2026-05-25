from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    video_path: Path
    frames_dir: Path
    output_dir: Path
    segmentation_weights: Path
    classification_weights: Path | None = None
    device: str = "cpu"
    frame_rate_fps: float | None = None
    pixel_size_um: float | None = None
    run_id: str = "bubbleid-run"
    confidence_threshold: float = 0.5

    @property
    def extension(self) -> str:
        return self.run_id

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["extension"] = self.extension
        return data


def _path_from(config_path: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    required = ["video_path", "frames_dir", "output_dir", "segmentation_weights"]
    missing = [key for key in required if not raw.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config fields: {joined}")

    return AnalysisConfig(
        video_path=_path_from(config_path, raw["video_path"]),
        frames_dir=_path_from(config_path, raw["frames_dir"]),
        output_dir=_path_from(config_path, raw["output_dir"]),
        segmentation_weights=_path_from(config_path, raw["segmentation_weights"]),
        classification_weights=_path_from(config_path, raw.get("classification_weights")),
        device=raw.get("device", "cpu"),
        frame_rate_fps=raw.get("frame_rate_fps"),
        pixel_size_um=raw.get("pixel_size_um"),
        run_id=raw.get("run_id", "bubbleid-run"),
        confidence_threshold=raw.get("confidence_threshold", 0.5),
    )
