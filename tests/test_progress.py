"""The optional progress hooks on run_eval and run_comparison."""

import asyncio

from evaluations_reference.compare import run_comparison
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
    async def score(self, case: TestCase, output: str, rubric: str) -> JudgeScores:
        return JudgeScores(relevance=8, faithfulness=8, reasoning="ok")


class FakeCandidate:
    async def generate(self, prompt: str, case: TestCase) -> str:
        return "out"


def _code_ds(n: int) -> EvalDataset:
    return EvalDataset(
        name="t",
        lens_config=LensConfig(enabled=[Lens.CODE], checks=[CheckSpec(name="length_bounds")]),
        test_cases=[TestCase(input=f"q{i}", expected="a") for i in range(n)],
    )


def test_run_eval_progress_non_judge() -> None:
    calls: list[tuple[int, int]] = []
    asyncio.run(run_eval(_code_ds(3), progress=lambda d, t: calls.append((d, t))))
    assert calls[-1] == (3, 3)


def test_run_eval_progress_judge() -> None:
    calls: list[tuple[int, int]] = []
    ds = EvalDataset(
        name="t",
        rubric="r",
        lens_config=LensConfig(enabled=[Lens.JUDGE]),
        test_cases=[TestCase(input="q1"), TestCase(input="q2")],
    )
    asyncio.run(run_eval(ds, judge=FakeJudge(), progress=lambda d, t: calls.append((d, t))))
    assert len(calls) == 2
    assert calls[-1] == (2, 2)


def test_run_comparison_progress() -> None:
    calls: list[tuple[int, int]] = []
    ds = _code_ds(2)
    asyncio.run(
        run_comparison(ds, "PA", "PB", FakeCandidate(), progress=lambda d, t: calls.append((d, t)))
    )
    # 2 cases x 2 prompts = 4 generations
    assert calls[-1] == (4, 4)
