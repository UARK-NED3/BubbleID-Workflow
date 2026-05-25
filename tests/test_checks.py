from pathlib import Path

from bubbleid_agent.checks import check_project
from bubbleid_agent.config import AnalysisConfig


def test_check_project_flags_low_frame_rate_for_tracking(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    video = tmp_path / "clip.avi"
    video.write_bytes(b"not a real video")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    config = AnalysisConfig(
        video_path=video,
        frames_dir=frames,
        output_dir=tmp_path / "out",
        segmentation_weights=weights,
        frame_rate_fps=150,
        pixel_size_um=None,
    )

    result = check_project(config)

    messages = [issue.message for issue in result.issues]
    assert any("tracking" in message and "600 fps" in message for message in messages)
    assert any(issue.code == "missing_calibration" for issue in result.issues)
    assert result.has_errors is False
