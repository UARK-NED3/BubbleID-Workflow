from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ImageSegmentationResult:
    image: str
    width_px: int
    height_px: int
    total_pixels: int
    bubble_mask_pixels: int
    vapor_fraction: float
    bubble_count: int
    mean_score: float | str
    max_score: float | str
    threshold: float
    weights: str


@dataclass(frozen=True)
class SegmentImagesResult:
    csv_path: Path
    summary_path: Path
    overlay_dir: Path
    mask_dir: Path
    rows: list[ImageSegmentationResult]


def vapor_fraction_from_masks(masks: np.ndarray, height: int, width: int) -> tuple[int, float]:
    if masks.size == 0 or masks.shape[0] == 0:
        return 0, 0.0
    combined = np.any(masks.astype(bool), axis=0)
    vapor_pixels = int(combined.sum())
    return vapor_pixels, vapor_pixels / float(height * width)


def write_results_csv(rows: Iterable[ImageSegmentationResult], csv_path: str | Path) -> Path:
    path = Path(csv_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [field.name for field in ImageSegmentationResult.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def _image_paths(images_dir: Path) -> list[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(images_dir.glob(pattern))
    return sorted(paths)


def _build_predictor(weights: Path, threshold: float, device: str, num_classes: int):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(weights.parent)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return DefaultPredictor(cfg)


def segment_images(
    images_dir: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    threshold: float = 0.4,
    device: str = "cpu",
    num_classes: int = 1,
    save_masks: bool = True,
    save_overlays: bool = True,
) -> SegmentImagesResult:
    import cv2

    image_dir = Path(images_dir).expanduser().resolve()
    weights_path = Path(weights).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    paths = _image_paths(image_dir)
    if not paths:
        raise ValueError(f"No supported image files found in {image_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")

    predictor = _build_predictor(weights_path, threshold, device, num_classes)
    rows: list[ImageSegmentationResult] = []
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        if len(instances) > 0:
            masks = instances.pred_masks.numpy().astype(bool)
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
            combined = np.any(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=bool)
            scores = np.array([])
            boxes = np.empty((0, 4))
            combined = np.zeros((height, width), dtype=bool)

        vapor_pixels, vapor_fraction = vapor_fraction_from_masks(masks, height, width)
        mask_u8 = (combined.astype(np.uint8) * 255)
        if save_masks:
            cv2.imwrite(str(mask_dir / f"{image_path.stem}_mask.png"), mask_u8)
        if save_overlays:
            overlay = image.copy()
            color_layer = np.zeros_like(overlay)
            color_layer[:, :, 2] = mask_u8
            overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(value) for value in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if idx < len(scores):
                    cv2.putText(overlay, f"{scores[idx]:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(str(overlay_dir / f"{image_path.stem}_overlay.jpg"), overlay)

        rows.append(
            ImageSegmentationResult(
                image=image_path.name,
                width_px=width,
                height_px=height,
                total_pixels=height * width,
                bubble_mask_pixels=vapor_pixels,
                vapor_fraction=vapor_fraction,
                bubble_count=int(len(instances)),
                mean_score=float(scores.mean()) if len(scores) else "",
                max_score=float(scores.max()) if len(scores) else "",
                threshold=threshold,
                weights=str(weights_path),
            )
        )

    csv_path = write_results_csv(rows, out_dir / "vapor_fraction_results.csv")
    vapor_fractions = [row.vapor_fraction for row in rows]
    summary = {
        "image_count": len(rows),
        "threshold": threshold,
        "device": device,
        "num_classes": num_classes,
        "weights": str(weights_path),
        "mean_vapor_fraction": float(np.mean(vapor_fractions)),
        "min_vapor_fraction": float(np.min(vapor_fractions)),
        "max_vapor_fraction": float(np.max(vapor_fractions)),
        "csv": str(csv_path),
        "overlays": str(overlay_dir),
        "masks": str(mask_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return SegmentImagesResult(csv_path=csv_path, summary_path=summary_path, overlay_dir=overlay_dir, mask_dir=mask_dir, rows=rows)
