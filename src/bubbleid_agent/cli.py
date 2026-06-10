from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checks import check_project
from .config import load_config
from .image_case import create_image_case_interactively, load_image_case, run_image_case
from .init_case import create_config_interactively
from .inspector import inspect_outputs
from .reporter import write_report
from .runner import run_analysis
from .segment_images import segment_images


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_check_project(args: argparse.Namespace) -> int:
    result = check_project(load_config(args.config))
    _print_json({"has_errors": result.has_errors, "issues": [issue.__dict__ for issue in result.issues]})
    return 1 if result.has_errors else 0


def cmd_run_analysis(args: argparse.Namespace) -> int:
    result = run_analysis(load_config(args.config), dry_run=args.dry_run)
    _print_json({"manifest_path": result.manifest_path, "invoked_bubbleid": result.invoked_bubbleid})
    return 0


def cmd_inspect_outputs(args: argparse.Namespace) -> int:
    report = inspect_outputs(args.output_dir, args.extension)
    _print_json(
        {
            "output_dir": report.output_dir,
            "extension": report.extension,
            "summary": report.summary,
            "findings": [finding.__dict__ for finding in report.findings],
        }
    )
    return 1 if any(finding.severity == "error" for finding in report.findings) else 0


def cmd_write_report(args: argparse.Namespace) -> int:
    inspection = inspect_outputs(args.output_dir, args.extension)
    write_report(args.manifest, inspection, args.report, use_openai=not args.offline, model=args.model)
    _print_json({"report_path": Path(args.report).resolve()})
    return 0


def cmd_init_case(args: argparse.Namespace) -> int:
    create_config_interactively(args.config)
    return 0


def cmd_segment_images(args: argparse.Namespace) -> int:
    result = segment_images(
        args.images_dir,
        args.weights,
        args.output_dir,
        threshold=args.threshold,
        device=args.device,
        num_classes=args.num_classes,
        save_masks=not args.no_masks,
        save_overlays=not args.no_overlays,
        filter_substrate=not args.no_substrate_filter,
        substrate_filter_strength=args.substrate_filter_strength,
    )
    _print_json(
        {
            "csv_path": result.csv_path,
            "summary_path": result.summary_path,
            "overlay_dir": result.overlay_dir,
            "mask_dir": result.mask_dir,
            "image_count": len(result.rows),
        }
    )
    return 0


def cmd_init_image_case(args: argparse.Namespace) -> int:
    create_image_case_interactively(args.config)
    return 0


def cmd_run_image_case(args: argparse.Namespace) -> int:
    result = run_image_case(load_image_case(args.config))
    _print_json(
        {
            "csv_path": result.csv_path,
            "summary_path": result.summary_path,
            "overlay_dir": result.overlay_dir,
            "mask_dir": result.mask_dir,
            "image_count": len(result.rows),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bubbleid-workflow", description="Reproducible workflow toolkit for BubbleID boiling image analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-case", help="Interactively create a BubbleID workflow config file.")
    init.add_argument("config", help="Destination JSON config path.")
    init.set_defaults(func=cmd_init_case)

    segment = subparsers.add_parser("segment-images", help="Segment still images and compute vapor fraction.")
    segment.add_argument("images_dir", help="Directory containing boiling images.")
    segment.add_argument("weights", help="Detectron2/BubbleID instance-segmentation weights.")
    segment.add_argument("output_dir", help="Directory for CSV, masks, overlays, and summary JSON.")
    segment.add_argument("--threshold", type=float, default=0.4, help="Mask R-CNN score threshold.")
    segment.add_argument("--device", default="cpu", help="Detectron2 device, usually cpu or cuda.")
    segment.add_argument("--num-classes", type=int, default=1, help="Number of instance-segmentation classes.")
    segment.add_argument("--no-masks", action="store_true", help="Do not write binary mask PNG files.")
    segment.add_argument("--no-overlays", action="store_true", help="Do not write overlay JPG files.")
    segment.add_argument("--no-substrate-filter", action="store_true", help="Keep raw model masks without removing neutral lower-substrate pixels.")
    segment.add_argument(
        "--substrate-filter-strength",
        choices=["aggressive", "conservative"],
        default="aggressive",
        help="Substrate cleanup strength. Aggressive removes more of the black slab but may remove some lower bubble pixels.",
    )
    segment.set_defaults(func=cmd_segment_images)

    init_image = subparsers.add_parser("init-image-case", help="Interactively create a still-image analysis config.")
    init_image.add_argument("config", help="Destination image-case JSON config path.")
    init_image.set_defaults(func=cmd_init_image_case)

    run_image = subparsers.add_parser("run-image-case", help="Run a still-image analysis config.")
    run_image.add_argument("config", help="Path to an image-case JSON config.")
    run_image.set_defaults(func=cmd_run_image_case)

    check = subparsers.add_parser("check-project", help="Validate inputs before running BubbleID.")
    check.add_argument("config", help="Path to a BubbleID workflow JSON config.")
    check.set_defaults(func=cmd_check_project)

    run = subparsers.add_parser("run-analysis", help="Run BubbleID and write a manifest.")
    run.add_argument("config", help="Path to a BubbleID workflow JSON config.")
    run.add_argument("--dry-run", action="store_true", help="Write the manifest without invoking BubbleID.")
    run.set_defaults(func=cmd_run_analysis)

    inspect = subparsers.add_parser("inspect-outputs", help="Inspect BubbleID output files.")
    inspect.add_argument("output_dir", help="BubbleID output directory.")
    inspect.add_argument("--extension", required=True, help="Run extension/run_id used by BubbleID.")
    inspect.set_defaults(func=cmd_inspect_outputs)

    report = subparsers.add_parser("write-report", help="Write a Markdown report from a manifest and output inspection.")
    report.add_argument("manifest", help="Path to manifest JSON.")
    report.add_argument("output_dir", help="BubbleID output directory.")
    report.add_argument("report", help="Destination Markdown report path.")
    report.add_argument("--extension", required=True, help="Run extension/run_id used by BubbleID.")
    report.add_argument("--model", help="OpenAI model for report drafting.")
    report.add_argument("--offline", action="store_true", help="Use deterministic local report generation.")
    report.set_defaults(func=cmd_write_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
