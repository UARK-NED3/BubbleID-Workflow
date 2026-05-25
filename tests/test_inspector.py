from pathlib import Path

from bubbleid_agent.inspector import inspect_outputs


def test_inspect_outputs_flags_missing_expected_files(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "vapor_case-a.npy").write_bytes(b"numpy")

    report = inspect_outputs(output, extension="case-a")

    codes = {finding.code for finding in report.findings}
    assert "missing_bounding_boxes" in codes
    assert "missing_bubble_size" in codes
    assert report.summary["files_found"] == 1
