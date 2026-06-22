"""Gradio UI: app instantiation smoke test and tab handlers (backends mocked)."""

from pathlib import Path

import gradio as gr
import pytest

from evaluations_reference import ui
from evaluations_reference.bootstrap import GeneratedCase
from evaluations_reference.errors import DatasetValidationError
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    Lens,
    LensConfig,
    TestCase,
)


class DummyProgress(gr.Progress):
    """Stand-in for gr.Progress when calling handlers outside a request."""

    def __call__(self, frac, desc=None, **kwargs):  # type: ignore[override]
        return None


class FakeCandidate:
    async def generate(self, prompt: str, case: TestCase) -> str:
        return "yes" if "yes" in prompt.lower() else "no"


class FakeGenerator:
    def __init__(self, n: int) -> None:
        self.n = n

    async def generate(self, description: str, count: int) -> list[GeneratedCase]:
        return [GeneratedCase(input=f"i{i}", expected=f"e{i}") for i in range(self.n)]


def _code_dataset(tmp_path: Path) -> str:
    path = tmp_path / "ds.json"
    EvalDataset(
        name="uitest",
        lens_config=LensConfig(
            enabled=[Lens.CODE],
            checks=[CheckSpec(name="keyword_present", params={"keywords": ["yes"]})],
        ),
        test_cases=[TestCase(input="q1"), TestCase(input="q2")],
    ).save(path)
    return str(path)


def test_build_app_instantiates() -> None:
    assert isinstance(ui.build_app(), gr.Blocks)


def test_compare_fn_and_drilldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "AnthropicCandidate", lambda model: FakeCandidate())
    ds = _code_dataset(tmp_path)
    summary, rows, _detail, path, report = ui.compare_fn(
        ds, "", "respond yes", "respond no", "m", progress=DummyProgress()
    )
    assert "Verdict" in summary
    assert len(rows) == 2
    assert path is not None and Path(path).exists()
    assert report is not None
    assert report.a_wins == 2

    detail = ui.compare_detail_fn(report, gr.SelectData(None, {"index": [0, 0], "value": None}))
    assert "Lens" in detail and "code" in detail


def test_compare_fn_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = _code_dataset(tmp_path)
    summary, _rows, _detail, _path, report = ui.compare_fn(
        ds, "", "a", "b", "m", progress=DummyProgress()
    )
    assert summary.startswith("⚠️")
    assert report is None


def test_compare_fn_requires_both_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "AnthropicCandidate", lambda model: FakeCandidate())
    ds = _code_dataset(tmp_path)
    summary, *_ = ui.compare_fn(ds, "", "", "b", "m", progress=DummyProgress())
    assert summary.startswith("⚠️")


def test_compare_detail_none_report() -> None:
    assert ui.compare_detail_fn(None, gr.SelectData(None, {"index": [0, 0], "value": None})) == ""


def test_bootstrap_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "AnthropicGenerator", lambda model: FakeGenerator(3))
    info, rows, path = ui.bootstrap_fn("invoices", 3, "", "m", progress=DummyProgress())
    assert "3" in info
    assert len(rows) == 3
    assert path is not None
    assert EvalDataset.load(path).lens_config.enabled == [Lens.CODE]


def test_bootstrap_fn_with_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "AnthropicGenerator", lambda model: FakeGenerator(2))
    _info, _rows, path = ui.bootstrap_fn("qa", 2, "score it", "m", progress=DummyProgress())
    assert path is not None
    ds = EvalDataset.load(path)
    assert ds.lens_config.enabled == [Lens.JUDGE]
    assert ds.rubric == "score it"


def test_run_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "AnthropicCandidate", lambda model: FakeCandidate())
    ds = _code_dataset(tmp_path)
    summary, rows, path = ui.run_fn(ds, "", "respond yes", "m", progress=DummyProgress())
    assert "uitest" in summary
    assert len(rows) == 2
    assert path is not None and Path(path).exists()


def test_parse_dataset_from_text() -> None:
    ds = ui._parse_dataset(None, '{"name": "x", "test_cases": [{"input": "q"}]}')
    assert ds.name == "x"


def test_parse_dataset_errors() -> None:
    with pytest.raises(DatasetValidationError):
        ui._parse_dataset(None, None)
    with pytest.raises(DatasetValidationError):
        ui._parse_dataset(None, "just a string")
