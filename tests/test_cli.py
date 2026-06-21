"""CLI coverage for `evals run` and `evals describe` (code lens only — no network)."""

import json
from pathlib import Path

from evaluations_reference.cli import main
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    Lens,
    LensConfig,
    TestCase,
)


def _write_dataset(path: Path) -> None:
    EvalDataset(
        name="cli-test",
        description="desc",
        lens_config=LensConfig(
            enabled=[Lens.CODE], checks=[CheckSpec(name="length_bounds", params={"min": 1})]
        ),
        test_cases=[TestCase(input="q", expected="an answer")],
    ).save(path)


def test_run_writes_report(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds.json"
    _write_dataset(ds_path)
    rc = main(["run", str(ds_path)])
    assert rc == 0
    report_path = tmp_path / "ds.report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["dataset_name"] == "cli-test"
    assert report["case_count"] == 1
    assert report["lens_averages"]["code"] == 1.0


def test_describe(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds.json"
    _write_dataset(ds_path)
    assert main(["describe", str(ds_path)]) == 0


def test_run_missing_file_returns_1() -> None:
    assert main(["run", "/nonexistent/path/ds.json"]) == 1


def test_run_empty_dataset_exits_0(tmp_path: Path) -> None:
    ds_path = tmp_path / "empty.json"
    EvalDataset(name="empty", lens_config=LensConfig(enabled=[Lens.CODE])).save(ds_path)
    assert main(["run", str(ds_path)]) == 0
    assert not (tmp_path / "empty.report.json").exists()
