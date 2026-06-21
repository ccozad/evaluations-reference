"""Evaluation runner.

Runs an :class:`EvalDataset` through its enabled lenses and aggregates the
results into a single :class:`EvalReport`. The runner is async so the
model-as-judge calls fan out concurrently across test cases.

The output graded for each test case comes from an optional ``task`` callable
(``input -> output``). When none is supplied — as in ``evals run`` — the test
case's ``expected`` reference answer is graded, which lets a dataset be sanity
checked on its own references. M2 supplies a real ``task`` to compare prompts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from evaluations_reference.errors import EmptyDatasetWarning
from evaluations_reference.lenses import code_checks, human
from evaluations_reference.lenses.judge import AnthropicJudge, ModelJudge
from evaluations_reference.models import (
    EvalDataset,
    EvalReport,
    Lens,
    LensResult,
    TestCase,
    TestCaseResult,
)

Task = Callable[[TestCase], str]


def _output_for(case: TestCase, task: Task | None) -> str:
    if task is not None:
        return task(case)
    return case.expected or ""


def _code_lens(dataset: EvalDataset, case: TestCase, output: str) -> LensResult:
    specs = dataset.lens_config.checks
    results = [code_checks.run_check(s.name, case, output, s.params) for s in specs]
    if not results:
        return LensResult(lens=Lens.CODE, score=None, note="no checks configured")
    score = sum(r.score for r in results) / len(results)
    return LensResult(
        lens=Lens.CODE,
        score=score,
        detail={"checks": [r.model_dump() for r in results]},
    )


async def _judge_lens(judge: ModelJudge, case: TestCase, output: str, rubric: str) -> LensResult:
    scores = await judge.score(case, output, rubric)
    # Two 0-10 sub-scores normalized to a single 0-1 lens score.
    score = (scores.relevance + scores.faithfulness) / 20.0
    return LensResult(
        lens=Lens.JUDGE, score=score, note=scores.reasoning, detail=scores.model_dump()
    )


def _human_lens(case: TestCase, output: str) -> LensResult:
    skip, reason = human.should_skip()
    if skip:
        return LensResult(lens=Lens.HUMAN, score=None, note=f"skipped: {reason}")
    rating = human.prompt_rating(case, output)
    # 1-5 rating normalized to 0-1 so the worst rating maps to 0.0.
    return LensResult(lens=Lens.HUMAN, score=(rating - 1) / 4.0, detail={"rating": rating})


def _average(scores: list[float | None]) -> float | None:
    present = [s for s in scores if s is not None]
    if not present:
        return None
    return sum(present) / len(present)


async def run_eval(
    dataset: EvalDataset,
    task: Task | None = None,
    judge: ModelJudge | None = None,
) -> EvalReport:
    """Run ``dataset`` through its enabled lenses and aggregate the results.

    Raises :class:`EmptyDatasetWarning` if the dataset has no test cases. If the
    judge lens is enabled and no ``judge`` is supplied, a default
    :class:`AnthropicJudge` is constructed (which requires ``ANTHROPIC_API_KEY``).
    """
    if not dataset.test_cases:
        raise EmptyDatasetWarning(f"dataset {dataset.name!r} has no test cases")

    enabled = dataset.lens_config.enabled
    # Construct the judge up front so a missing API key fails before any grading.
    if Lens.JUDGE in enabled and judge is None:
        judge = AnthropicJudge()

    outputs = [_output_for(case, task) for case in dataset.test_cases]

    # Judge calls fan out concurrently across all test cases.
    judge_results: list[LensResult | None]
    if Lens.JUDGE in enabled:
        assert judge is not None
        judge_results = await asyncio.gather(
            *(
                _judge_lens(judge, case, output, dataset.rubric)
                for case, output in zip(dataset.test_cases, outputs, strict=True)
            )
        )
    else:
        judge_results = [None] * len(dataset.test_cases)

    results: list[TestCaseResult] = []
    for case, output, judged in zip(dataset.test_cases, outputs, judge_results, strict=True):
        lens_results: list[LensResult] = []
        if Lens.CODE in enabled:
            lens_results.append(_code_lens(dataset, case, output))
        if judged is not None:
            lens_results.append(judged)
        if Lens.HUMAN in enabled:
            lens_results.append(_human_lens(case, output))
        results.append(TestCaseResult(test_case=case, output=output, lens_results=lens_results))

    lens_averages: dict[Lens, float | None] = {}
    for lens in enabled:
        lens_averages[lens] = _average(
            [lr.score for r in results for lr in r.lens_results if lr.lens == lens]
        )

    return EvalReport(
        dataset_name=dataset.name,
        case_count=len(dataset.test_cases),
        lens_averages=lens_averages,
        results=results,
    )
