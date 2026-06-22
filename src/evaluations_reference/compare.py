"""Side-by-side comparison of two prompt variants against the same eval set.

Each variant runs every test case through the candidate model to produce an
output, the three lenses grade both, and the results are diffed: per-lens score
deltas, per-test-case winners, and an overall verdict.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from evaluations_reference.candidate import CandidateModel
from evaluations_reference.errors import EmptyDatasetWarning
from evaluations_reference.lenses.judge import AnthropicJudge, ModelJudge
from evaluations_reference.models import (
    EvalDataset,
    EvalReport,
    Lens,
    LensResult,
    TestCase,
)
from evaluations_reference.runner import ProgressFn, run_eval

# Score difference below which a per-case result is a tie.
TIE_EPSILON = 1e-9


class LensDelta(BaseModel):
    """Per-lens average scores for both variants and their difference (a - b)."""

    lens: Lens
    avg_a: float | None
    avg_b: float | None
    delta: float | None


class CaseLensComparison(BaseModel):
    """One lens's verdict for a test case under both variants (for drill-down)."""

    lens: Lens
    score_a: float | None
    note_a: str
    score_b: float | None
    note_b: str


class CaseComparison(BaseModel):
    """One test case's aggregate score under each variant and the winner."""

    index: int
    input: str
    score_a: float | None
    score_b: float | None
    delta: float | None
    winner: str  # "a", "b", or "tie"
    lenses: list[CaseLensComparison] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    """The full side-by-side comparison of two prompt variants."""

    dataset_name: str
    label_a: str
    label_b: str
    case_count: int
    overall_a: float | None
    overall_b: float | None
    a_wins: int
    b_wins: int
    ties: int
    verdict: str
    lens_deltas: list[LensDelta] = Field(default_factory=list)
    cases: list[CaseComparison] = Field(default_factory=list)


def _case_score(lens_results: list[LensResult]) -> float | None:
    """Aggregate a case's lens scores (mean of those that produced a score)."""
    present = [lr.score for lr in lens_results if lr.score is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _case_lens_comparisons(
    a_results: list[LensResult], b_results: list[LensResult]
) -> list[CaseLensComparison]:
    """Pair each lens's A and B verdicts for one test case."""
    b_by_lens = {lr.lens: lr for lr in b_results}
    out: list[CaseLensComparison] = []
    for ar in a_results:
        br = b_by_lens.get(ar.lens)
        out.append(
            CaseLensComparison(
                lens=ar.lens,
                score_a=ar.score,
                note_a=ar.note,
                score_b=br.score if br else None,
                note_b=br.note if br else "",
            )
        )
    return out


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _winner(score_a: float | None, score_b: float | None) -> str:
    a = score_a if score_a is not None else 0.0
    b = score_b if score_b is not None else 0.0
    if a - b > TIE_EPSILON:
        return "a"
    if b - a > TIE_EPSILON:
        return "b"
    return "tie"


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def build_comparison(
    dataset_name: str,
    report_a: EvalReport,
    report_b: EvalReport,
    label_a: str,
    label_b: str,
) -> ComparisonReport:
    """Diff two :class:`EvalReport` results into a :class:`ComparisonReport`."""
    cases: list[CaseComparison] = []
    for index, (ra, rb) in enumerate(zip(report_a.results, report_b.results, strict=True)):
        score_a = _case_score(ra.lens_results)
        score_b = _case_score(rb.lens_results)
        cases.append(
            CaseComparison(
                index=index,
                input=ra.test_case.input,
                score_a=score_a,
                score_b=score_b,
                delta=_delta(score_a, score_b),
                winner=_winner(score_a, score_b),
                lenses=_case_lens_comparisons(ra.lens_results, rb.lens_results),
            )
        )

    a_wins = sum(1 for c in cases if c.winner == "a")
    b_wins = sum(1 for c in cases if c.winner == "b")
    ties = sum(1 for c in cases if c.winner == "tie")

    overall_a = _mean([c.score_a for c in cases])
    overall_b = _mean([c.score_b for c in cases])

    lenses = list(dict.fromkeys([*report_a.lens_averages, *report_b.lens_averages]))
    lens_deltas = [
        LensDelta(
            lens=lens,
            avg_a=report_a.lens_averages.get(lens),
            avg_b=report_b.lens_averages.get(lens),
            delta=_delta(report_a.lens_averages.get(lens), report_b.lens_averages.get(lens)),
        )
        for lens in lenses
    ]

    return ComparisonReport(
        dataset_name=dataset_name,
        label_a=label_a,
        label_b=label_b,
        case_count=len(cases),
        overall_a=overall_a,
        overall_b=overall_b,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        verdict=_verdict(label_a, label_b, overall_a, overall_b),
        lens_deltas=lens_deltas,
        cases=cases,
    )


def _verdict(label_a: str, label_b: str, overall_a: float | None, overall_b: float | None) -> str:
    a = overall_a if overall_a is not None else 0.0
    b = overall_b if overall_b is not None else 0.0
    if a - b > TIE_EPSILON:
        return f"{label_a} wins (+{a - b:.3f})"
    if b - a > TIE_EPSILON:
        return f"{label_b} wins (+{b - a:.3f})"
    return "tie"


async def _generate_all(
    candidate: CandidateModel,
    prompt: str,
    cases: Sequence[TestCase],
    on_done: Callable[[], None] | None = None,
) -> list[str]:
    async def _one(case: TestCase) -> str:
        out = await candidate.generate(prompt, case)
        if on_done is not None:
            on_done()
        return out

    return list(await asyncio.gather(*(_one(c) for c in cases)))


def _make_task(cases: Sequence[TestCase], outputs: Sequence[str]):
    # Map by object identity: the runner passes these same TestCase objects back.
    by_id = {id(c): o for c, o in zip(cases, outputs, strict=True)}
    return lambda case: by_id[id(case)]


async def run_comparison(
    dataset: EvalDataset,
    prompt_a: str,
    prompt_b: str,
    candidate: CandidateModel,
    judge: ModelJudge | None = None,
    label_a: str = "A",
    label_b: str = "B",
    progress: ProgressFn | None = None,
) -> ComparisonReport:
    """Run both prompt variants against ``dataset`` and diff the results.

    Outputs for both variants are generated concurrently, then each variant is
    graded by the dataset's enabled lenses. ``progress`` is an optional hook
    called ``(completed, total)`` as the 2N candidate generations complete.
    """
    if not dataset.test_cases:
        raise EmptyDatasetWarning(f"dataset {dataset.name!r} has no test cases")

    if Lens.JUDGE in dataset.lens_config.enabled and judge is None:
        judge = AnthropicJudge()

    gen_total = 2 * len(dataset.test_cases)
    gen_done = 0

    def _tick() -> None:
        nonlocal gen_done
        gen_done += 1
        if progress is not None:
            progress(gen_done, gen_total)

    outputs_a, outputs_b = await asyncio.gather(
        _generate_all(candidate, prompt_a, dataset.test_cases, _tick),
        _generate_all(candidate, prompt_b, dataset.test_cases, _tick),
    )

    report_a, report_b = await asyncio.gather(
        run_eval(dataset, task=_make_task(dataset.test_cases, outputs_a), judge=judge),
        run_eval(dataset, task=_make_task(dataset.test_cases, outputs_b), judge=judge),
    )

    return build_comparison(dataset.name, report_a, report_b, label_a, label_b)
