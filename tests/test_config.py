from pathlib import Path

from bubbleid_agent.config import load_config


def test_load_config_normalizes_paths(tmp_path: Path):
    frames = tmp_path / "frames"
    video = tmp_path / "clip.avi"
    output = tmp_path / "outputs"
    weights = tmp_path / "weights.pth"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f"""{{
  "video_path": "{video.as_posix()}",
  "frames_dir": "{frames.as_posix()}",
  "output_dir": "{output.as_posix()}",
  "segmentation_weights": "{weights.as_posix()}",
  "device": "cpu",
  "frame_rate_fps": 3000,
  "pixel_size_um": 12.5,
  "run_id": "boiling-test"
}}""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.video_path == video
    assert config.frames_dir == frames
    assert config.output_dir == output
    assert config.segmentation_weights == weights
    assert config.extension == "boiling-test"
    assert config.to_manifest()["frame_rate_fps"] == 3000
