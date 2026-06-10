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


def substrate_pixel_mask(
    image_bgr: np.ndarray,
    lower_fraction: float = 0.45,
    max_chroma: int = 45,
    max_intensity: int = 60,
) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3).")
    height = image_bgr.shape[0]
    image_i16 = image_bgr.astype(np.int16)
    chroma = image_i16.max(axis=2) - image_i16.min(axis=2)
    intensity = image_i16.mean(axis=2)
    lower_rows = np.arange(height)[:, None] >= int(height * lower_fraction)
    return lower_rows & (chroma <= max_chroma) & (intensity <= max_intensity)


def surface_type_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "flat" in lowered:
        return "flat"
    if "_mp" in lowered or " mp" in lowered or "-mp" in lowered:
        return "mp"
    if "_mc" in lowered or " mc" in lowered or "-mc" in lowered:
        return "mc"
    return None


def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes: list[list[float]] = []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    if not boxes:
        return np.empty((0, 4))
    return np.array(boxes, dtype=float)


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        components = [
            (int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]), int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]), int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, count)
        ]
        return labels, components
    except ImportError:
        labels = np.zeros(binary.shape, dtype=np.int32)
        components: list[tuple[int, int, int, int, int]] = []
        current = 0
        height, width = binary.shape
        for start_y, start_x in np.argwhere(binary):
            if labels[start_y, start_x]:
                continue
            current += 1
            stack = [(int(start_y), int(start_x))]
            labels[start_y, start_x] = current
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if binary[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
            components.append((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(xs)))
    return labels, components


def substrate_reference_mask(reference_bgr: np.ndarray, lower_fraction: float = 0.45, max_intensity: int = 190) -> np.ndarray:
    if reference_bgr.ndim != 3 or reference_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR reference image with shape (height, width, 3).")
    height = reference_bgr.shape[0]
    gray = reference_bgr.astype(np.int16).mean(axis=2)
    lower_rows = np.arange(height)[:, None] >= int(height * lower_fraction)
    candidate = lower_rows & (gray <= max_intensity)
    labels, components = _connected_components(candidate)
    substrate = np.zeros_like(candidate, dtype=bool)
    min_area = max(3, int(height * reference_bgr.shape[1] * 0.002))
    for label, (_x, y, _width, component_height, area) in enumerate(components, start=1):
        touches_bottom = y + component_height >= height - 1
        if touches_bottom and area >= min_area:
            substrate |= labels == label
    return substrate


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    except ImportError:
        y_idx = (np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
        x_idx = (np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
        return mask[np.ix_(y_idx, x_idx)]


def load_substrate_reference_masks(reference_dir: str | Path, target_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required to load substrate reference images.") from exc

    root = Path(reference_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Substrate reference directory does not exist: {root}")
    masks: dict[str, np.ndarray] = {}
    for path in _image_paths(root):
        surface = surface_type_from_name(path.name)
        if surface is None:
            continue
        reference = cv2.imread(str(path))
        if reference is None:
            raise ValueError(f"Could not read substrate reference image: {path}")
        masks[surface] = _resize_bool_mask(substrate_reference_mask(reference), target_shape)
    if not masks:
        raise ValueError(f"No Flat/MP/MC substrate reference images found in {root}")
    return masks


def _slab_like_substrate(mask: np.ndarray, substrate: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)
    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    mask_bottom = int(ys.max())
    candidate = mask & substrate
    labels, components = _connected_components(candidate)
    remove = np.zeros_like(mask, dtype=bool)
    for label, (x, y, width, height, area) in enumerate(components, start=1):
        if area == 0:
            continue
        component_bottom = y + height - 1
        near_bottom = component_bottom >= mask_bottom - max(3, int(mask_height * 0.08))
        wide = width >= max(2, int(mask_width * 0.25))
        flat = width >= height * 1.8
        not_tall = height <= max(8, int(mask_height * 0.35))
        if near_bottom and wide and flat and not_tall:
            remove |= labels == label
    return remove


def _remove_reference_substrate_components(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    labels, components = _connected_components(mask)
    filtered = mask.copy()
    image_area = mask.shape[0] * mask.shape[1]
    max_component_area = max(100, int(image_area * 0.025))
    min_y = int(mask.shape[0] * 0.62)
    for label, (_x, y, _width, _height, area) in enumerate(components, start=1):
        if area == 0 or area > max_component_area or y < min_y:
            continue
        component = labels == label
        reference_overlap = float((component & reference_mask).sum()) / float(area)
        if reference_overlap >= 0.45:
            filtered[component] = False
    return filtered


def remove_substrate_from_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    image_bgr: np.ndarray,
    strength: str = "aggressive",
    reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if masks.size == 0 or masks.shape[0] == 0:
        height, width = image_bgr.shape[:2]
        return np.zeros((0, height, width), dtype=bool), np.array([]), np.empty((0, 4))
    filtered_masks = masks.astype(bool).copy()
    if strength == "conservative":
        substrate = substrate_pixel_mask(image_bgr)
        for idx, mask in enumerate(filtered_masks):
            filtered_masks[idx] = mask & ~_slab_like_substrate(mask, substrate)
    elif strength == "aggressive":
        substrate = substrate_pixel_mask(image_bgr, lower_fraction=0.62, max_chroma=35, max_intensity=95)
        filtered_masks &= ~substrate
    else:
        raise ValueError(f"Unknown substrate filter strength: {strength}")
    if reference_mask is not None:
        reference_bool = reference_mask.astype(bool)
        for idx, mask in enumerate(filtered_masks):
            filtered_masks[idx] = _remove_reference_substrate_components(mask, reference_bool)
    keep = filtered_masks.reshape(filtered_masks.shape[0], -1).sum(axis=1) > 0
    filtered_masks = filtered_masks[keep]
    filtered_scores = scores[keep] if len(scores) == len(keep) else scores
    boxes = _boxes_from_masks(filtered_masks)
    return filtered_masks, filtered_scores, boxes


def apply_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int] = (35, 35, 220), alpha: float = 0.5) -> np.ndarray:
    overlay = image_bgr.copy()
    if mask.any():
        mask_bool = mask.astype(bool)
        color = np.array(color_bgr, dtype=np.float32)
        pixels = overlay[mask_bool].astype(np.float32)
        overlay[mask_bool] = np.clip((1.0 - alpha) * pixels + alpha * color, 0, 255).astype(np.uint8)
    return overlay


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
    filter_substrate: bool = True,
    substrate_filter_strength: str = "aggressive",
    substrate_references_dir: str | Path | None = None,
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
    reference_masks: dict[str, np.ndarray] = {}
    if substrate_references_dir is not None:
        first_image = cv2.imread(str(paths[0]))
        if first_image is None:
            raise ValueError(f"Could not read image: {paths[0]}")
        reference_masks = load_substrate_reference_masks(substrate_references_dir, first_image.shape[:2])
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
        else:
            masks = np.zeros((0, height, width), dtype=bool)
            scores = np.array([])
            boxes = np.empty((0, 4))
        if filter_substrate:
            reference_mask = reference_masks.get(surface_type_from_name(image_path.name)) if reference_masks else None
            masks, scores, boxes = remove_substrate_from_masks(masks, scores, image, strength=substrate_filter_strength, reference_mask=reference_mask)
        combined = np.any(masks, axis=0) if len(masks) else np.zeros((height, width), dtype=bool)

        vapor_pixels, vapor_fraction = vapor_fraction_from_masks(masks, height, width)
        mask_u8 = (combined.astype(np.uint8) * 255)
        if save_masks:
            cv2.imwrite(str(mask_dir / f"{image_path.stem}_mask.png"), mask_u8)
        if save_overlays:
            overlay = apply_mask_overlay(image, combined)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (35, 35, 220), 2)
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
                bubble_count=int(len(masks)),
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
        "filter_substrate": filter_substrate,
        "substrate_filter_strength": substrate_filter_strength if filter_substrate else "none",
        "substrate_references_dir": str(Path(substrate_references_dir).expanduser().resolve()) if substrate_references_dir is not None else "",
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
