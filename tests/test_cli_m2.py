"""CLI coverage for `evals compare` and `evals bootstrap` (backends mocked)."""

import json
from pathlib import Path

import pytest

from evaluations_reference.bootstrap import GeneratedCase
from evaluations_reference.cli import main
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    Lens,
    LensConfig,
    TestCase,
)


class FakeCandidate:
    async def generate(self, prompt: str, case: TestCase) -> str:
        return "yes" if "yes" in prompt.lower() else "no"


class FakeGenerator:
    def __init__(self, n: int) -> None:
        self.n = n

    async def generate(self, description: str, count: int) -> list[GeneratedCase]:
        return [GeneratedCase(input=f"i{i}", expected=f"e{i}") for i in range(self.n)]


def test_cli_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluations_reference.cli.AnthropicCandidate", lambda model: FakeCandidate()
    )

    ds_path = tmp_path / "ds.json"
    EvalDataset(
        name="cmp",
        lens_config=LensConfig(
            enabled=[Lens.CODE],
            checks=[CheckSpec(name="keyword_present", params={"keywords": ["yes"]})],
        ),
        test_cases=[TestCase(input="q1"), TestCase(input="q2")],
    ).save(ds_path)
    (tmp_path / "a.txt").write_text("respond yes")
    (tmp_path / "b.txt").write_text("respond no")

    rc = main(
        [
            "compare",
            str(ds_path),
            "--prompt-a",
            str(tmp_path / "a.txt"),
            "--prompt-b",
            str(tmp_path / "b.txt"),
        ]
    )
    assert rc == 0
    report = json.loads((tmp_path / "ds.comparison.json").read_text())
    assert report["overall_a"] == 1.0
    assert report["overall_b"] == 0.0
    assert report["a_wins"] == 2


def test_cli_compare_missing_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluations_reference.cli.AnthropicCandidate", lambda model: FakeCandidate()
    )
    ds_path = tmp_path / "ds.json"
    EvalDataset(
        name="cmp", lens_config=LensConfig(enabled=[Lens.CODE]), test_cases=[TestCase(input="q")]
    ).save(ds_path)
    rc = main(["compare", str(ds_path), "--prompt-a", "/nope/a.txt", "--prompt-b", "/nope/b.txt"])
    assert rc == 1


def test_cli_bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluations_reference.cli.AnthropicGenerator", lambda model: FakeGenerator(3)
    )
    out = tmp_path / "boot.json"
    rc = main(["bootstrap", "--description", "invoice tests", "--count", "3", "--output", str(out)])
    assert rc == 0
    ds = EvalDataset.load(out)
    assert len(ds.test_cases) == 3
    assert ds.lens_config.enabled == [Lens.CODE]


def test_cli_bootstrap_with_rubric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluations_reference.cli.AnthropicGenerator", lambda model: FakeGenerator(2)
    )
    out = tmp_path / "boot.yaml"
    rc = main(
        [
            "bootstrap",
            "--description",
            "qa",
            "--count",
            "2",
            "--output",
            str(out),
            "--rubric",
            "score relevance",
        ]
    )
    assert rc == 0
    ds = EvalDataset.load(out)
    assert ds.lens_config.enabled == [Lens.JUDGE]
    assert ds.rubric == "score relevance"
