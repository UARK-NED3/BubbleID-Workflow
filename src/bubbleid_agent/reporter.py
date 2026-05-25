from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .inspector import InspectionReport


def _offline_report(manifest: dict, inspection: InspectionReport) -> str:
    findings = "\n".join(f"- {item.severity.upper()}: {item.message}" for item in inspection.findings) or "- No inspection findings."
    return (
        "# BubbleID Analysis Report\n\n"
        f"**Run ID:** {manifest.get('run_id', inspection.extension)}\n\n"
        "## Methods\n\n"
        "BubbleID-Agent prepared a reproducible analysis summary from the run manifest and output inspection. "
        "BubbleID remains the source of segmentation, classification, tracking, vapor-fraction, departure, and interface-velocity outputs.\n\n"
        "## Run Metadata\n\n"
        f"- Frame rate: {manifest.get('frame_rate_fps', 'not provided')} fps\n"
        f"- Pixel size: {manifest.get('pixel_size_um', 'not provided')} um/pixel\n"
        f"- Device: {manifest.get('device', 'not provided')}\n"
        f"- Output files found: {inspection.summary.get('files_found', 0)} of {inspection.summary.get('expected_files', 0)}\n\n"
        "## Quality Checks\n\n"
        f"{findings}\n\n"
        "## Interpretation Notes\n\n"
        "Review segmentation masks, tracking continuity, calibration, and frame rate before using dynamic quantities for boiling physics claims.\n"
    )


def _openai_report(manifest: dict, inspection: InspectionReport, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = {
        "manifest": manifest,
        "inspection": {
            "extension": inspection.extension,
            "summary": inspection.summary,
            "findings": [finding.__dict__ for finding in inspection.findings],
        },
    }
    response = client.responses.create(
        model=model,
        instructions=(
            "Write a concise Markdown report for a boiling heat-transfer research lab. "
            "Do not invent results. Distinguish BubbleID outputs from quality-control warnings."
        ),
        input=json.dumps(prompt, indent=2),
    )
    return response.output_text


def write_report(
    manifest_path: str | Path,
    inspection: InspectionReport,
    output_path: str | Path,
    use_openai: bool = True,
    model: str | None = None,
) -> str:
    load_dotenv()
    manifest_file = Path(manifest_path).expanduser().resolve()
    report_file = Path(output_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    selected_model = model or os.getenv("BUBBLEID_AGENT_MODEL", "gpt-5")
    if use_openai and os.getenv("OPENAI_API_KEY"):
        content = _openai_report(manifest, inspection, selected_model)
    else:
        content = _offline_report(manifest, inspection)

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(content, encoding="utf-8")
    return content
