from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from .segment_images import SegmentImagesResult, segment_images


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass(frozen=True)
class ImageCaseConfig:
    images_dir: Path
    weights: Path
    output_dir: Path
    threshold: float = 0.4
    device: str = "cpu"
    num_classes: int = 1
    save_masks: bool = True
    save_overlays: bool = True
    filter_substrate: bool = True

    def write(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        for key in ["images_dir", "weights", "output_dir"]:
            data[key] = str(data[key]).replace("\\", "/")
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return target


def _resolve_from(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_image_case(path: str | Path) -> ImageCaseConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [key for key in ["images_dir", "weights", "output_dir"] if not raw.get(key)]
    if missing:
        raise ValueError(f"Missing required image-case fields: {', '.join(missing)}")
    return ImageCaseConfig(
        images_dir=_resolve_from(config_path, raw["images_dir"]),
        weights=_resolve_from(config_path, raw["weights"]),
        output_dir=_resolve_from(config_path, raw["output_dir"]),
        threshold=float(raw.get("threshold", 0.4)),
        device=raw.get("device", "cpu"),
        num_classes=int(raw.get("num_classes", 1)),
        save_masks=bool(raw.get("save_masks", True)),
        save_overlays=bool(raw.get("save_overlays", True)),
        filter_substrate=bool(raw.get("filter_substrate", True)),
    )


def _ask(input_fn: InputFn, prompt: str, default: str | None = None) -> str:
    label = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    answer = input_fn(label).strip()
    if answer:
        return answer
    return default or ""


def create_image_case_interactively(
    config_path: str | Path,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    target = Path(config_path).expanduser().resolve()
    output_fn("BubbleID Workflow image-case setup")
    output_fn("Use this for still images or independently sampled frames.")
    images_dir = _ask(input_fn, "Image folder")
    weights = _ask(input_fn, "BubbleID instance-segmentation weights")
    output_dir = _ask(input_fn, "Output folder", "outputs/image-case")
    threshold = float(_ask(input_fn, "Detection threshold", "0.4"))
    device = _ask(input_fn, "Device", "cpu")
    num_classes = int(_ask(input_fn, "Number of segmentation classes", "1"))
    filter_substrate = _ask(input_fn, "Remove dark lower substrate from masks", "yes").lower() in {"y", "yes", "true", "1"}
    config = ImageCaseConfig(
        images_dir=Path(images_dir),
        weights=Path(weights),
        output_dir=Path(output_dir),
        threshold=threshold,
        device=device,
        num_classes=num_classes,
        filter_substrate=filter_substrate,
    )
    written = config.write(target)
    output_fn(f"Wrote image-case config: {written}")
    output_fn(f"Next: bubbleid-workflow run-image-case {written}")
    return written


def run_image_case(config: ImageCaseConfig, segment_fn=segment_images) -> SegmentImagesResult:
    return segment_fn(
        images_dir=config.images_dir,
        weights=config.weights,
        output_dir=config.output_dir,
        threshold=config.threshold,
        device=config.device,
        num_classes=config.num_classes,
        save_masks=config.save_masks,
        save_overlays=config.save_overlays,
        filter_substrate=config.filter_substrate,
    )
