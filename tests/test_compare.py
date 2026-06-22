"""Comparison math and the run_comparison flow (with a mocked candidate)."""

import asyncio

from evaluations_reference.compare import build_comparison, run_comparison
from evaluations_reference.lenses.judge import JudgeScores
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    EvalReport,
    Lens,
    LensConfig,
    LensResult,
    TestCase,
    TestCaseResult,
)


def _report(scores: list[float], name: str = "ds") -> EvalReport:
    """Build an EvalReport with one code-lens result per score."""
    results = [
        TestCaseResult(
            test_case=TestCase(input=f"q{i}"),
            output=f"o{i}",
            lens_results=[LensResult(lens=Lens.CODE, score=s)],
        )
        for i, s in enumerate(scores)
    ]
    avg = sum(scores) / len(scores)
    return EvalReport(
        dataset_name=name, case_count=len(scores), lens_averages={Lens.CODE: avg}, results=results
    )


def test_a_always_wins() -> None:
    cmp = build_comparison("ds", _report([1.0, 1.0, 1.0]), _report([0.0, 0.0, 0.0]), "A", "B")
    assert (cmp.a_wins, cmp.b_wins, cmp.ties) == (3, 0, 0)
    assert cmp.verdict.startswith("A wins")
    assert cmp.lens_deltas[0].delta == 1.0


def test_b_always_wins() -> None:
    cmp = build_comparison("ds", _report([0.0, 0.0]), _report([1.0, 1.0]), "A", "B")
    assert (cmp.a_wins, cmp.b_wins, cmp.ties) == (0, 2, 0)
    assert cmp.verdict.startswith("B wins")


def test_mixed_wins() -> None:
    cmp = build_comparison("ds", _report([1.0, 0.0, 0.5]), _report([0.0, 1.0, 0.5]), "A", "B")
    assert (cmp.a_wins, cmp.b_wins, cmp.ties) == (1, 1, 1)
    # overall averages are equal -> tie verdict
    assert cmp.verdict == "tie"


def test_all_ties() -> None:
    cmp = build_comparison("ds", _report([0.5, 0.5]), _report([0.5, 0.5]), "A", "B")
    assert (cmp.a_wins, cmp.b_wins, cmp.ties) == (0, 0, 2)
    assert cmp.verdict == "tie"


class FakeCandidate:
    """Returns a fixed per-prompt output so comparisons are deterministic."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    async def generate(self, prompt: str, case: TestCase) -> str:
        return self.mapping[prompt]


def test_run_comparison_end_to_end_code_lens() -> None:
    ds = EvalDataset(
        name="cmp",
        lens_config=LensConfig(
            enabled=[Lens.CODE],
            checks=[CheckSpec(name="keyword_present", params={"keywords": ["yes"]})],
        ),
        test_cases=[TestCase(input="q1"), TestCase(input="q2")],
    )
    # Prompt A always answers "yes" (passes), B always "no" (fails).
    candidate = FakeCandidate({"PA": "yes", "PB": "no"})
    cmp = asyncio.run(run_comparison(ds, "PA", "PB", candidate, label_a="PA", label_b="PB"))
    assert cmp.overall_a == 1.0
    assert cmp.overall_b == 0.0
    assert cmp.a_wins == 2
    assert cmp.verdict.startswith("PA wins")


class FakeJudge:
    async def score(self, case: TestCase, output: str, rubric: str) -> JudgeScores:
        # Score by output content so A and B differ.
        rel = 10 if output == "good" else 2
        return JudgeScores(relevance=rel, faithfulness=rel, reasoning="r")


def test_run_comparison_with_judge_injected() -> None:
    ds = EvalDataset(
        name="cmp",
        rubric="r",
        lens_config=LensConfig(enabled=[Lens.JUDGE]),
        test_cases=[TestCase(input="q1")],
    )
    candidate = FakeCandidate({"PA": "good", "PB": "bad"})
    cmp = asyncio.run(
        run_comparison(ds, "PA", "PB", candidate, judge=FakeJudge(), label_a="PA", label_b="PB")
    )
    assert cmp.overall_a == 1.0  # (10+10)/20
    assert cmp.overall_b == 0.2  # (2+2)/20
    assert cmp.a_wins == 1
