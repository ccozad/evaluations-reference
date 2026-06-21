"""Command-line interface for the eval framework.

``evals run <dataset>`` runs the full eval set, renders a terminal table, and
writes a JSON report next to the dataset. ``evals describe <dataset>`` summarizes
a dataset without running it.

Logging and human-readable tables go to stderr; stdout is reserved for
machine-readable output (the report path on ``run``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pydantic
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from evaluations_reference.errors import (
    DatasetValidationError,
    EmptyDatasetWarning,
    EvalError,
)
from evaluations_reference.models import EvalDataset, EvalReport, Lens
from evaluations_reference.runner import run_eval

logger = logging.getLogger("evaluations_reference")
# Human-readable output goes to stderr; stdout stays clean for machine consumers.
_err_console = Console(stderr=True)


def _load_dataset(path_str: str) -> EvalDataset:
    path = Path(path_str)
    if not path.exists():
        raise DatasetValidationError(f"dataset file not found: {path}")
    try:
        return EvalDataset.load(path)
    except pydantic.ValidationError as exc:
        raise DatasetValidationError(f"malformed dataset {path}: {exc}") from exc
    except (ValueError, OSError) as exc:
        raise DatasetValidationError(f"could not parse dataset {path}: {exc}") from exc


def _report_path(dataset_path: str) -> Path:
    p = Path(dataset_path)
    return p.with_suffix(".report.json")


def _render_table(report: EvalReport) -> None:
    table = Table(title=f"Eval report — {report.dataset_name} ({report.case_count} cases)")
    table.add_column("Lens")
    table.add_column("Average score", justify="right")
    for lens, avg in report.lens_averages.items():
        table.add_row(lens.value, "—" if avg is None else f"{avg:.3f}")
    _err_console.print(table)


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    try:
        report = asyncio.run(run_eval(dataset))
    except EmptyDatasetWarning as exc:
        logger.warning("%s; nothing to evaluate", exc)
        return 0

    out_path = _report_path(args.dataset)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _render_table(report)
    # Machine-readable: the report path on stdout.
    print(out_path)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    console = Console()
    console.print(f"[bold]{dataset.name}[/bold]")
    if dataset.description:
        console.print(dataset.description)
    console.print(f"Test cases: {len(dataset.test_cases)}")
    enabled = ", ".join(lens.value for lens in dataset.lens_config.enabled) or "(none)"
    console.print(f"Enabled lenses: {enabled}")
    if Lens.CODE in dataset.lens_config.enabled:
        checks = ", ".join(c.name for c in dataset.lens_config.checks) or "(none)"
        console.print(f"Code checks: {checks}")
    console.print(f"Rubric: {dataset.rubric or '(none)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Load a local .env (e.g. ANTHROPIC_API_KEY) for local runs, searching from
    # the current working directory upward. Does not override variables already
    # set in the environment.
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(prog="evals", description="Three-lens LLM eval harness.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging on stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a dataset and write a JSON report")
    run_p.add_argument("dataset", help="path to a .json or .yaml dataset")
    run_p.set_defaults(func=_cmd_run)

    desc_p = sub.add_parser("describe", help="summarize a dataset without running it")
    desc_p.add_argument("dataset", help="path to a .json or .yaml dataset")
    desc_p.set_defaults(func=_cmd_describe)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return args.func(args)
    except EvalError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
