from __future__ import annotations

from dataclasses import dataclass

from .config import AnalysisConfig


@dataclass(frozen=True)
class CheckIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class CheckResult:
    issues: list[CheckIssue]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def check_project(config: AnalysisConfig) -> CheckResult:
    issues: list[CheckIssue] = []

    if not config.video_path.exists():
        issues.append(CheckIssue("error", "missing_video", f"Video file does not exist: {config.video_path}"))
    if not config.frames_dir.exists():
        issues.append(CheckIssue("error", "missing_frames_dir", f"Frames directory does not exist: {config.frames_dir}"))
    if not config.segmentation_weights.exists():
        issues.append(
            CheckIssue("error", "missing_segmentation_weights", f"Segmentation weights do not exist: {config.segmentation_weights}")
        )
    if config.classification_weights and not config.classification_weights.exists():
        issues.append(
            CheckIssue("warning", "missing_classification_weights", f"Classification weights do not exist: {config.classification_weights}")
        )

    if config.frame_rate_fps is None:
        issues.append(CheckIssue("warning", "missing_frame_rate", "Frame rate is missing; tracking and velocity checks cannot be assessed."))
    elif config.frame_rate_fps < 600:
        issues.append(
            CheckIssue(
                "warning",
                "low_frame_rate_for_tracking",
                "Frame rate is below 600 fps; BubbleID tracking, departure, and interface velocity outputs may be unreliable.",
            )
        )

    if config.pixel_size_um is None:
        issues.append(
            CheckIssue(
                "warning",
                "missing_calibration",
                "Pixel calibration is missing; pixel-scale outputs cannot be converted to physical bubble sizes or velocities.",
            )
        )

    if config.device not in {"cpu", "gpu", "cuda"}:
        issues.append(CheckIssue("warning", "unknown_device", f"Device should usually be cpu, gpu, or cuda; got {config.device!r}."))

    if not issues:
        issues.append(CheckIssue("info", "ready", "Project inputs are present and ready for BubbleID analysis."))

    return CheckResult(issues)
