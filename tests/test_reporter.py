from pathlib import Path

from bubbleid_agent.inspector import InspectionReport
from bubbleid_agent.reporter import write_report


def test_write_report_offline_fallback_contains_key_sections(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"run_id": "case-a", "frame_rate_fps": 3000}', encoding="utf-8")
    inspection = InspectionReport(output_dir=tmp_path, extension="case-a", findings=[], summary={"files_found": 3})
    report_path = tmp_path / "report.md"

    content = write_report(manifest, inspection, report_path, use_openai=False)

    assert "BubbleID Analysis Report" in content
    assert "case-a" in content
    assert report_path.read_text(encoding="utf-8") == content
