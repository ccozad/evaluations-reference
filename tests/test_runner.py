"""Runner coverage: aggregation math, judge happy path (mocked), human skip, error contract."""

import asyncio

import pytest

from evaluations_reference.errors import EmptyDatasetWarning, MissingAPIKeyError
from evaluations_reference.lenses.judge import JudgeScores
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    Lens,
    LensConfig,
    TestCase,
)
from evaluations_reference.runner import run_eval


class FakeJudge:
    """A stand-in ModelJudge that returns fixed scores — no network."""

    def __init__(self, relevance: int = 8, faithfulness: int = 6) -> None:
        self.relevance = relevance
        self.faithfulness = faithfulness
        self.calls = 0

    async def score(self, case: TestCase, output: str, rubric: str) -> JudgeScores:
        self.calls += 1
        return JudgeScores(
            relevance=self.relevance, faithfulness=self.faithfulness, reasoning="fixed"
        )


def _dataset(enabled: list[Lens], checks: list[CheckSpec] | None = None) -> EvalDataset:
    return EvalDataset(
        name="t",
        rubric="r",
        lens_config=LensConfig(enabled=enabled, checks=checks or []),
        test_cases=[
            TestCase(input="q1", expected="Paris is the capital."),
            TestCase(input="q2", expected="Red is a color."),
        ],
    )


def test_code_lens_aggregation_math() -> None:
    # Two checks: length passes (1.0), keyword_absent fails on case 1 only.
    ds = _dataset(
        [Lens.CODE],
        checks=[
            CheckSpec(name="length_bounds", params={"min": 1}),
            CheckSpec(name="keyword_absent", params={"keywords": ["Paris"]}),
        ],
    )
    report = asyncio.run(run_eval(ds))
    # Case 1: (1.0 + 0.0)/2 = 0.5 ; Case 2: (1.0 + 1.0)/2 = 1.0 ; avg = 0.75
    assert report.results[0].lens_results[0].score == 0.5
    assert report.results[1].lens_results[0].score == 1.0
    assert report.lens_averages[Lens.CODE] == 0.75


def test_judge_happy_path_mocked() -> None:
    ds = _dataset([Lens.JUDGE])
    judge = FakeJudge(relevance=8, faithfulness=6)
    report = asyncio.run(run_eval(ds, judge=judge))
    assert judge.calls == 2  # one per test case
    # (8 + 6) / 20 = 0.7
    for result in report.results:
        assert result.lens_results[0].lens == Lens.JUDGE
        assert result.lens_results[0].score == pytest.approx(0.7)
    assert report.lens_averages[Lens.JUDGE] == pytest.approx(0.7)


def test_human_lens_auto_skips() -> None:
    # conftest sets SKIP_HUMAN_EVAL=1, so the human lens records None + a note.
    ds = _dataset([Lens.HUMAN])
    report = asyncio.run(run_eval(ds))
    human_result = report.results[0].lens_results[0]
    assert human_result.lens == Lens.HUMAN
    assert human_result.score is None
    assert "skipped" in human_result.note
    assert report.lens_averages[Lens.HUMAN] is None


def test_all_three_lenses_together() -> None:
    ds = _dataset([Lens.CODE, Lens.JUDGE, Lens.HUMAN], checks=[CheckSpec(name="length_bounds")])
    report = asyncio.run(run_eval(ds, judge=FakeJudge()))
    lenses = {lr.lens for lr in report.results[0].lens_results}
    assert lenses == {Lens.CODE, Lens.JUDGE, Lens.HUMAN}


def test_empty_dataset_raises() -> None:
    ds = EvalDataset(name="empty", lens_config=LensConfig(enabled=[Lens.CODE]))
    with pytest.raises(EmptyDatasetWarning):
        asyncio.run(run_eval(ds))


def test_judge_enabled_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = _dataset([Lens.JUDGE])
    with pytest.raises(MissingAPIKeyError):
        asyncio.run(run_eval(ds))
