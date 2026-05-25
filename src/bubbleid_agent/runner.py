from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

from .config import AnalysisConfig


@dataclass(frozen=True)
class RunResult:
    manifest_path: Path
    invoked_bubbleid: bool


def _write_manifest(config: AnalysisConfig, invoked: bool) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = config.to_manifest()
    manifest["bubbleid_invoked"] = invoked
    manifest_path = config.output_dir / f"manifest_{config.extension}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _load_bubbleid_class() -> Any:
    module = importlib.import_module("BubbleID.BubbleID")
    return getattr(module, "DataAnalysis")


def run_analysis(config: AnalysisConfig, dry_run: bool = False) -> RunResult:
    if dry_run:
        return RunResult(manifest_path=_write_manifest(config, invoked=False), invoked_bubbleid=False)

    data_analysis_class = _load_bubbleid_class()
    analysis = data_analysis_class(
        str(config.frames_dir),
        str(config.video_path),
        str(config.output_dir),
        config.extension,
        str(config.segmentation_weights),
        config.device,
    )
    analysis.GenerateData(thres=config.confidence_threshold)
    return RunResult(manifest_path=_write_manifest(config, invoked=True), invoked_bubbleid=True)
