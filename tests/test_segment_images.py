import csv
from pathlib import Path

import numpy as np

from bubbleid_agent.segment_images import (
    ImageSegmentationResult,
    remove_substrate_from_masks,
    substrate_pixel_mask,
    vapor_fraction_from_masks,
    write_results_csv,
)


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


def test_substrate_pixel_mask_finds_dark_neutral_lower_slab():
    image = np.full((6, 6, 3), 245, dtype=np.uint8)
    image[4:, :, :] = [45, 45, 45]
    image[4:, 1:3, :] = [40, 40, 190]

    substrate = substrate_pixel_mask(image)

    assert substrate[4, 0]
    assert substrate[5, 5]
    assert not substrate[2, 0]
    assert not substrate[4, 1]


def test_remove_substrate_from_masks_keeps_bubble_and_removes_slab():
    image = np.full((6, 6, 3), 245, dtype=np.uint8)
    image[4:, :, :] = [45, 45, 45]
    image[1:4, 2:4, :] = [35, 35, 190]
    mask = np.zeros((1, 6, 6), dtype=bool)
    mask[0, 1:5, 2:4] = True
    scores = np.array([0.92])

    filtered_masks, filtered_scores, boxes = remove_substrate_from_masks(mask, scores, image)

    assert filtered_scores.tolist() == [0.92]
    assert filtered_masks.shape == (1, 6, 6)
    assert filtered_masks[0, 1:4, 2:4].all()
    assert not filtered_masks[0, 4, 2]
    assert boxes.tolist() == [[2.0, 1.0, 4.0, 4.0]]
