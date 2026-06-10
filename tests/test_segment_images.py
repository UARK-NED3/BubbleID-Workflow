import csv
from pathlib import Path

import numpy as np

from bubbleid_agent.segment_images import (
    ImageSegmentationResult,
    apply_mask_overlay,
    remove_substrate_from_masks,
    substrate_reference_mask,
    substrate_pixel_mask,
    surface_type_from_name,
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


def test_surface_type_from_name_matches_demo_filenames():
    assert surface_type_from_name("2_Flat Cu_10 kPa_CHF_B-285_Frame-35831.jpg") == "flat"
    assert surface_type_from_name("17_MP Cu_100 kPa_CHF_B-319_Frame-12678.jpg") == "mp"
    assert surface_type_from_name("25_MC Cu_100 kPa_ONB_B-372_Frame-3083.jpg") == "mc"
    assert surface_type_from_name("unknown.jpg") is None


def test_substrate_reference_mask_keeps_bottom_surface_not_background_band():
    image = np.full((10, 10, 3), 240, dtype=np.uint8)
    image[4:6, :, :] = [40, 40, 40]
    image[7:, 2:8, :] = [70, 70, 70]

    substrate = substrate_reference_mask(image)

    assert not substrate[4, 5]
    assert substrate[9, 5]
    assert not substrate[9, 0]


def test_conservative_substrate_filter_keeps_bubble_and_removes_slab():
    image = np.full((6, 6, 3), 245, dtype=np.uint8)
    image[4:, :, :] = [45, 45, 45]
    image[1:4, 2:4, :] = [35, 35, 190]
    mask = np.zeros((1, 6, 6), dtype=bool)
    mask[0, 1:5, 2:4] = True
    scores = np.array([0.92])

    filtered_masks, filtered_scores, boxes = remove_substrate_from_masks(mask, scores, image, strength="conservative")

    assert filtered_scores.tolist() == [0.92]
    assert filtered_masks.shape == (1, 6, 6)
    assert filtered_masks[0, 1:4, 2:4].all()
    assert not filtered_masks[0, 4, 2]
    assert boxes.tolist() == [[2.0, 1.0, 4.0, 4.0]]


def test_aggressive_substrate_filter_removes_lower_neutral_band():
    image = np.full((10, 10, 3), 245, dtype=np.uint8)
    image[7:, :, :] = [80, 80, 80]
    image[2:5, 4:6, :] = [35, 35, 190]
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 2:9, 2:8] = True
    scores = np.array([0.81])

    filtered_masks, _, _ = remove_substrate_from_masks(mask, scores, image, strength="aggressive")

    assert filtered_masks[0, 2:5, 4:6].all()
    assert not filtered_masks[0, 7, 4]


def test_reference_substrate_filter_removes_surface_template_region():
    image = np.full((8, 8, 3), 240, dtype=np.uint8)
    image[2:4, 3:5, :] = [35, 35, 190]
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 2:4, 3:5] = True
    mask[0, 6:8, 2:7] = True
    reference_mask = np.zeros((8, 8), dtype=bool)
    reference_mask[6:8, 1:7] = True

    filtered_masks, _, _ = remove_substrate_from_masks(
        mask,
        np.array([0.8]),
        image,
        strength="aggressive",
        reference_mask=reference_mask,
    )

    assert filtered_masks[0, 2:4, 3:5].all()
    assert not filtered_masks[0, 6, 3]


def test_apply_mask_overlay_uses_consistent_red_tint():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0, :] = [240, 240, 240]
    image[1, 1, :] = [20, 20, 20]
    mask = np.array([[True, False], [False, True]])

    overlay = apply_mask_overlay(image, mask, alpha=0.5)

    assert overlay[0, 0, 2] > overlay[0, 0, 1]
    assert overlay[1, 1, 2] > overlay[1, 1, 1]
    assert overlay[0, 1].tolist() == [0, 0, 0]
