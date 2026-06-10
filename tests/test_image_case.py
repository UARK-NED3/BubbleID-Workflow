import json
from pathlib import Path

from bubbleid_agent.image_case import ImageCaseConfig, create_image_case_interactively, load_image_case, run_image_case
from bubbleid_agent.segment_images import ImageSegmentationResult, SegmentImagesResult


def test_image_case_config_round_trips_paths(tmp_path: Path):
    config_path = tmp_path / "case.json"
    config = ImageCaseConfig(
        images_dir=Path("images"),
        weights=Path("weights/model_1class.pth"),
        output_dir=Path("outputs/case-a"),
        threshold=0.4,
        device="cpu",
        num_classes=1,
    )

    config.write(config_path)
    loaded = load_image_case(config_path)

    assert loaded.images_dir == (tmp_path / "images").resolve()
    assert loaded.weights == (tmp_path / "weights/model_1class.pth").resolve()
    assert loaded.output_dir == (tmp_path / "outputs/case-a").resolve()
    assert loaded.threshold == 0.4


def test_create_image_case_interactively_writes_student_friendly_config(tmp_path: Path):
    answers = iter(["images", "model_1class.pth", "outputs/demo", "", "", "", "", "", ""])

    config_path = create_image_case_interactively(
        tmp_path / "image-case.json",
        input_fn=lambda prompt: next(answers),
        output_fn=lambda message: None,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["images_dir"] == "images"
    assert data["weights"] == "model_1class.pth"
    assert data["output_dir"] == "outputs/demo"
    assert data["threshold"] == 0.4
    assert data["device"] == "cpu"
    assert data["num_classes"] == 1
    assert data["filter_substrate"] is True
    assert data["substrate_filter_strength"] == "aggressive"
    assert data["substrate_references_dir"] is None


def test_run_image_case_delegates_to_segment_images(tmp_path: Path):
    config = ImageCaseConfig(
        images_dir=tmp_path / "images",
        weights=tmp_path / "weights.pth",
        output_dir=tmp_path / "outputs",
        threshold=0.5,
        device="cpu",
        num_classes=1,
        filter_substrate=False,
        substrate_filter_strength="conservative",
        substrate_references_dir=tmp_path / "refs",
    )
    calls = {}

    def fake_segment_images(**kwargs):
        calls.update(kwargs)
        row = ImageSegmentationResult(
            image="case.jpg",
            width_px=2,
            height_px=2,
            total_pixels=4,
            bubble_mask_pixels=1,
            vapor_fraction=0.25,
            bubble_count=1,
            mean_score=0.9,
            max_score=0.9,
            threshold=0.5,
            weights=str(config.weights),
        )
        return SegmentImagesResult(
            csv_path=tmp_path / "outputs" / "vapor_fraction_results.csv",
            summary_path=tmp_path / "outputs" / "summary.json",
            overlay_dir=tmp_path / "outputs" / "overlays",
            mask_dir=tmp_path / "outputs" / "masks",
            rows=[row],
        )

    result = run_image_case(config, segment_fn=fake_segment_images)

    assert result.rows[0].vapor_fraction == 0.25
    assert calls["images_dir"] == config.images_dir
    assert calls["threshold"] == 0.5
    assert calls["filter_substrate"] is False
    assert calls["substrate_filter_strength"] == "conservative"
    assert calls["substrate_references_dir"] == tmp_path / "refs"
