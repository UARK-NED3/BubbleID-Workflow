from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class InspectionReport:
    output_dir: Path
    extension: str
    findings: list[Finding]
    summary: dict[str, int]


def inspect_outputs(output_dir: str | Path, extension: str) -> InspectionReport:
    output_path = Path(output_dir).expanduser().resolve()
    findings: list[Finding] = []
    expected = {
        "bounding_boxes": output_path / f"bb-Boiling-{extension}.txt",
        "model_outputs": output_path / f"bb-Boiling-output-{extension}.txt",
        "vapor_fraction": output_path / f"vapor_{extension}.npy",
        "bubble_size": output_path / f"bubble_size_bt-{extension}.npy",
        "bubble_ids": output_path / f"bubind_{extension}.npy",
        "frame_ids": output_path / f"frames_{extension}.npy",
    }

    files_found = 0
    for label, path in expected.items():
        if path.exists():
            files_found += 1
        else:
            findings.append(Finding("warning", f"missing_{label}", f"Expected BubbleID output is missing: {path.name}"))

    for text_file in [expected["bounding_boxes"], expected["model_outputs"]]:
        if text_file.exists() and text_file.stat().st_size == 0:
            findings.append(Finding("warning", "empty_output_file", f"Output file is empty: {text_file.name}"))

    if not output_path.exists():
        findings.append(Finding("error", "missing_output_dir", f"Output directory does not exist: {output_path}"))

    return InspectionReport(
        output_dir=output_path,
        extension=extension,
        findings=findings,
        summary={"files_found": files_found, "expected_files": len(expected)},
    )
