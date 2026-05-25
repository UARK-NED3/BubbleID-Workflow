from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _ask(input_fn: InputFn, prompt: str, default: str | None = None) -> str:
    label = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    answer = input_fn(label).strip()
    if answer:
        return answer
    if default is not None:
        return default
    return ""


def _optional_float(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def create_config_interactively(
    config_path: str | Path,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    target = Path(config_path).expanduser().resolve()

    output_fn("BubbleID-Agent case setup")
    output_fn("Enter paths relative to this config file, or absolute paths if preferred.")

    video_path = _ask(input_fn, "Boiling video path (.avi)")
    frames_dir = _ask(input_fn, "Extracted frames directory")
    output_dir = _ask(input_fn, "Output directory", "outputs/bubbleid-run")
    segmentation_weights = _ask(input_fn, "Segmentation model weights path")
    classification_weights = _ask(input_fn, "Classification weights path (optional)", "")
    device = _ask(input_fn, "Device", "cpu")
    frame_rate = _ask(input_fn, "Frame rate in fps (blank if unknown)", "")
    pixel_size = _ask(input_fn, "Pixel size in um/pixel (blank if unknown)", "")
    run_id = _ask(input_fn, "Run ID / output extension", "bubbleid-run")
    confidence = _ask(input_fn, "Confidence threshold", "0.5")

    config = {
        "video_path": video_path,
        "frames_dir": frames_dir,
        "output_dir": output_dir,
        "segmentation_weights": segmentation_weights,
        "classification_weights": classification_weights or None,
        "device": device,
        "frame_rate_fps": _optional_float(frame_rate),
        "pixel_size_um": _optional_float(pixel_size),
        "run_id": run_id,
        "confidence_threshold": float(confidence),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output_fn(f"Wrote config: {target}")
    output_fn(f"Next: bubbleid-agent check-project {target}")
    return target
