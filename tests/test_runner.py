import json
from pathlib import Path

from bubbleid_agent.config import AnalysisConfig
from bubbleid_agent.runner import run_analysis


def test_run_analysis_writes_manifest_without_invoking_bubbleid(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    video = tmp_path / "clip.avi"
    video.write_bytes(b"video")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "out"
    config = AnalysisConfig(
        video_path=video,
        frames_dir=frames,
        output_dir=output,
        segmentation_weights=weights,
        frame_rate_fps=3000,
        pixel_size_um=10.0,
        run_id="case-a",
    )

    result = run_analysis(config, dry_run=True)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.invoked_bubbleid is False
    assert manifest["run_id"] == "case-a"
    assert manifest["bubbleid_invoked"] is False
