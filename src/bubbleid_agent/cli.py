from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checks import check_project
from .config import load_config
from .init_case import create_config_interactively
from .inspector import inspect_outputs
from .reporter import write_report
from .runner import run_analysis


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bubbleid-agent", description="Workflow agent for BubbleID boiling image analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-case", help="Interactively create a BubbleID-Agent config file.")
    init.add_argument("config", help="Destination JSON config path.")
    init.set_defaults(func=cmd_init_case)

    check = subparsers.add_parser("check-project", help="Validate inputs before running BubbleID.")
    check.add_argument("config", help="Path to a BubbleID-Agent JSON config.")
    check.set_defaults(func=cmd_check_project)

    run = subparsers.add_parser("run-analysis", help="Run BubbleID and write a manifest.")
    run.add_argument("config", help="Path to a BubbleID-Agent JSON config.")
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
