import json
from pathlib import Path

from bubbleid_agent.init_case import create_config_interactively


def test_create_config_interactively_writes_expected_json(tmp_path: Path):
    answers = iter(
        [
            "data/case-a/clip.avi",
            "data/case-a/frames",
            "outputs/case-a",
            "weights/model_final.pth",
            "weights/classifier.pth",
            "gpu",
            "3000",
            "12.5",
            "case-a",
            "0.7",
        ]
    )
    messages: list[str] = []

    config_path = create_config_interactively(
        tmp_path / "case-a.json",
        input_fn=lambda prompt: next(answers),
        output_fn=messages.append,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data == {
        "video_path": "data/case-a/clip.avi",
        "frames_dir": "data/case-a/frames",
        "output_dir": "outputs/case-a",
        "segmentation_weights": "weights/model_final.pth",
        "classification_weights": "weights/classifier.pth",
        "device": "gpu",
        "frame_rate_fps": 3000.0,
        "pixel_size_um": 12.5,
        "run_id": "case-a",
        "confidence_threshold": 0.7,
    }
    assert any("Wrote config" in message for message in messages)


def test_create_config_interactively_applies_defaults_and_optional_values(tmp_path: Path):
    answers = iter(
        [
            "clip.avi",
            "frames",
            "",
            "model.pth",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    config_path = create_config_interactively(tmp_path / "bubble.json", input_fn=lambda prompt: next(answers), output_fn=lambda _: None)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["output_dir"] == "outputs/bubbleid-run"
    assert data["classification_weights"] is None
    assert data["device"] == "cpu"
    assert data["frame_rate_fps"] is None
    assert data["pixel_size_um"] is None
    assert data["run_id"] == "bubbleid-run"
    assert data["confidence_threshold"] == 0.5
