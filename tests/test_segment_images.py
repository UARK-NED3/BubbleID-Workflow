import csv
from pathlib import Path

import numpy as np

from bubbleid_agent.segment_images import ImageSegmentationResult, vapor_fraction_from_masks, write_results_csv


def test_vapor_fraction_from_masks_uses_union_not_sum():
    masks = np.array(
        [
            [[True, True], [False, False]],
            [[False, True], [True, False]],
        ]
    )

    vapor_pixels, vapor_fraction = vapor_fraction_from_masks(masks, height=2, width=2)

    assert vapor_pixels == 3
    assert vapor_fraction == 0.75


def test_vapor_fraction_from_empty_masks_is_zero():
    masks = np.zeros((0, 3, 4), dtype=bool)

    vapor_pixels, vapor_fraction = vapor_fraction_from_masks(masks, height=3, width=4)

    assert vapor_pixels == 0
    assert vapor_fraction == 0.0


def test_write_results_csv(tmp_path: Path):
    rows = [
        ImageSegmentationResult(
            image="case.jpg",
            width_px=4,
            height_px=3,
            total_pixels=12,
            bubble_mask_pixels=3,
            vapor_fraction=0.25,
            bubble_count=2,
            mean_score=0.8,
            max_score=0.9,
            threshold=0.4,
            weights="weights.pth",
        )
    ]

    csv_path = write_results_csv(rows, tmp_path / "results.csv")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))

    assert data[0]["image"] == "case.jpg"
    assert data[0]["vapor_fraction"] == "0.25"
