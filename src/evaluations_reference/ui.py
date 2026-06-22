"""Gradio browser UI — a thin three-tab layer over the M1/M2 library.

Tab 1 compares two prompt variants side by side, Tab 2 bootstraps a dataset from
a description, and Tab 3 grades a single prompt. Launch with ``evals ui`` or
``python -m evaluations_reference.ui``.
"""

from __future__ import annotations

import asyncio
import tempfile

import gradio as gr
import pydantic
import yaml

from evaluations_reference.bootstrap import (
    DEFAULT_BOOTSTRAP_MODEL,
    AnthropicGenerator,
    bootstrap_dataset,
)
from evaluations_reference.candidate import DEFAULT_CANDIDATE_MODEL, AnthropicCandidate
from evaluations_reference.compare import CaseComparison, ComparisonReport, run_comparison
from evaluations_reference.errors import DatasetValidationError, EvalError
from evaluations_reference.models import EvalDataset, EvalReport, Lens, TestCase
from evaluations_reference.runner import run_eval

_LENS_ORDER = [Lens.CODE, Lens.JUDGE, Lens.HUMAN]
# Gradio injects a fresh per-request Progress; the default is just a marker.
_DEFAULT_PROGRESS = gr.Progress()


# --- formatting helpers -----------------------------------------------------


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _fmt_signed(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def _cell(text: str) -> str:
    """Sanitize free text for a markdown table cell."""
    return text.replace("|", "/").replace("\n", " ")


def _parse_dataset(file_path: str | None, text: str | None) -> EvalDataset:
    """Load a dataset from an uploaded file or pasted JSON/YAML text."""
    try:
        if file_path:
            return EvalDataset.load(file_path)
        if text and text.strip():
            # YAML is a superset of JSON, so this parses either.
            return EvalDataset.model_validate(yaml.safe_load(text))
    except (pydantic.ValidationError, ValueError, OSError) as exc:
        raise DatasetValidationError(f"could not parse dataset: {exc}") from exc
    raise DatasetValidationError("Provide a dataset file or paste dataset text.")


def _tmp_file(suffix: str, prefix: str) -> str:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=suffix, prefix=prefix, delete=False, encoding="utf-8"
    )
    f.close()
    return f.name


def _comparison_summary_md(r: ComparisonReport) -> str:
    lines = [
        f"### Verdict: {r.verdict}",
        f"**A** = `{r.label_a}` · **B** = `{r.label_b}` · {r.case_count} cases",
        f"A wins: {r.a_wins} · B wins: {r.b_wins} · ties: {r.ties}",
        "",
        "| Lens | A | B | Δ (a-b) |",
        "|---|---|---|---|",
    ]
    for d in r.lens_deltas:
        lines.append(
            f"| {d.lens.value} | {_fmt(d.avg_a)} | {_fmt(d.avg_b)} | {_fmt_signed(d.delta)} |"
        )
    return "\n".join(lines)


def _case_detail_md(c: CaseComparison) -> str:
    lines = [
        f"#### Case {c.index}",
        f"> {_cell(c.input)}",
        "",
        "| Lens | A score | A verdict | B score | B verdict |",
        "|---|---|---|---|---|",
    ]
    for ls in c.lenses:
        lines.append(
            f"| {ls.lens.value} | {_fmt(ls.score_a)} | {_cell(ls.note_a)} "
            f"| {_fmt(ls.score_b)} | {_cell(ls.note_b)} |"
        )
    return "\n".join(lines)


def _comparison_rows(r: ComparisonReport) -> list[list[object]]:
    return [
        [c.index, c.input[:80], _fmt(c.score_a), _fmt(c.score_b), _fmt_signed(c.delta), c.winner]
        for c in r.cases
    ]


def _report_summary_md(report: EvalReport) -> str:
    lines = [
        f"### {report.dataset_name} — {report.case_count} cases",
        "",
        "| Lens | Average |",
        "|---|---|",
    ]
    for lens, avg in report.lens_averages.items():
        lines.append(f"| {lens.value} | {_fmt(avg)} |")
    return "\n".join(lines)


def _report_rows(report: EvalReport) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, res in enumerate(report.results):
        by_lens = {lr.lens: lr.score for lr in res.lens_results}
        rows.append(
            [i, res.test_case.input[:80], res.output[:80]]
            + [_fmt(by_lens.get(lens)) for lens in _LENS_ORDER]
        )
    return rows


# --- single-prompt run (UI-side orchestration of existing library pieces) ---


async def _run_single(
    dataset: EvalDataset, prompt: str, candidate: AnthropicCandidate, progress
) -> EvalReport:
    cases = dataset.test_cases
    outputs = await asyncio.gather(*(candidate.generate(prompt, c) for c in cases))
    by_id = {id(c): o for c, o in zip(cases, outputs, strict=True)}

    def task(case: TestCase) -> str:
        return by_id[id(case)]

    return await run_eval(dataset, task=task, progress=progress)


# --- tab handlers -----------------------------------------------------------


def compare_fn(
    dataset_file: str | None,
    dataset_text: str | None,
    prompt_a: str,
    prompt_b: str,
    model: str,
    progress: gr.Progress = _DEFAULT_PROGRESS,
):
    try:
        dataset = _parse_dataset(dataset_file, dataset_text)
        if not prompt_a.strip() or not prompt_b.strip():
            raise DatasetValidationError("Both Prompt A and Prompt B are required.")
        candidate = AnthropicCandidate(model=model or DEFAULT_CANDIDATE_MODEL)

        def cb(done: int, total: int) -> None:
            progress(done / total, desc=f"Generating outputs {done}/{total}")

        report = asyncio.run(run_comparison(dataset, prompt_a, prompt_b, candidate, progress=cb))
    except EvalError as exc:
        return f"⚠️ {exc}", [], "", None, None

    path = _tmp_file(".json", "comparison-")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return _comparison_summary_md(report), _comparison_rows(report), "", path, report


def compare_detail_fn(report: ComparisonReport | None, evt: gr.SelectData) -> str:
    if report is None:
        return ""
    index = evt.index
    row = index[0] if isinstance(index, (list, tuple)) else index
    if row is None or row >= len(report.cases):
        return ""
    return _case_detail_md(report.cases[row])


def bootstrap_fn(
    description: str,
    count: float,
    rubric: str,
    model: str,
    progress: gr.Progress = _DEFAULT_PROGRESS,
):
    try:
        progress(0.0, desc="Generating cases…")
        generator = AnthropicGenerator(model=model or DEFAULT_BOOTSTRAP_MODEL)
        dataset = asyncio.run(
            bootstrap_dataset(description, int(count), generator, rubric=rubric or None)
        )
        progress(1.0, desc="Done")
    except EvalError as exc:
        return f"⚠️ {exc}", [], None

    path = _tmp_file(".json", "dataset-")
    dataset.save(path)
    rows = [[i, tc.input[:80], (tc.expected or "")[:80]] for i, tc in enumerate(dataset.test_cases)]
    lenses = ", ".join(lens.value for lens in dataset.lens_config.enabled)
    info = f"Generated **{len(dataset.test_cases)}** cases · lenses: {lenses}"
    return info, rows, path


def run_fn(
    dataset_file: str | None,
    dataset_text: str | None,
    prompt: str,
    model: str,
    progress: gr.Progress = _DEFAULT_PROGRESS,
):
    try:
        dataset = _parse_dataset(dataset_file, dataset_text)
        if not prompt.strip():
            raise DatasetValidationError("A prompt is required.")
        candidate = AnthropicCandidate(model=model or DEFAULT_CANDIDATE_MODEL)

        def cb(done: int, total: int) -> None:
            progress(done / total, desc=f"Grading {done}/{total}")

        report = asyncio.run(_run_single(dataset, prompt, candidate, cb))
    except EvalError as exc:
        return f"⚠️ {exc}", [], None

    path = _tmp_file(".json", "report-")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return _report_summary_md(report), _report_rows(report), path


# --- app assembly -----------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the Gradio app with all three tabs."""
    with gr.Blocks(title="Evaluations Reference") as app:
        gr.Markdown(
            "# Evaluations Reference\n"
            "Three-lens LLM eval harness. Set `ANTHROPIC_API_KEY` in the "
            "environment before running — tabs that call the API will show a "
            "message if it is missing."
        )

        with gr.Tabs():
            with gr.Tab("Compare prompts"):
                with gr.Row():
                    c_file = gr.File(
                        label="Dataset file (JSON/YAML)", file_types=[".json", ".yaml", ".yml"]
                    )
                    c_text = gr.Textbox(label="…or paste dataset (JSON/YAML)", lines=6)
                c_prompt_a = gr.Textbox(label="Prompt A (system prompt)", lines=4)
                c_prompt_b = gr.Textbox(label="Prompt B (system prompt)", lines=4)
                c_model = gr.Textbox(label="Model", value=DEFAULT_CANDIDATE_MODEL)
                c_run = gr.Button("Run comparison", variant="primary")
                c_summary = gr.Markdown()
                c_df = gr.Dataframe(
                    headers=["#", "input", "A", "B", "Δ(a-b)", "winner"],
                    interactive=False,
                    label="Per-test-case (click a row for lens detail)",
                )
                c_detail = gr.Markdown()
                c_download = gr.File(label="Download report JSON")
                c_state = gr.State()
                c_run.click(
                    compare_fn,
                    inputs=[c_file, c_text, c_prompt_a, c_prompt_b, c_model],
                    outputs=[c_summary, c_df, c_detail, c_download, c_state],
                )
                c_df.select(compare_detail_fn, inputs=[c_state], outputs=[c_detail])

            with gr.Tab("Bootstrap dataset"):
                b_desc = gr.Textbox(label="Dataset description", lines=3)
                b_count = gr.Slider(
                    label="Test case count", minimum=1, maximum=50, value=10, step=1
                )
                b_rubric = gr.Textbox(label="Model-as-judge rubric (optional)", lines=3)
                b_model = gr.Textbox(label="Model", value=DEFAULT_BOOTSTRAP_MODEL)
                b_run = gr.Button("Generate", variant="primary")
                b_info = gr.Markdown()
                b_df = gr.Dataframe(
                    headers=["#", "input", "expected"], interactive=False, label="Generated cases"
                )
                b_download = gr.File(label="Download dataset JSON")
                b_run.click(
                    bootstrap_fn,
                    inputs=[b_desc, b_count, b_rubric, b_model],
                    outputs=[b_info, b_df, b_download],
                )

            with gr.Tab("Single run"):
                with gr.Row():
                    r_file = gr.File(
                        label="Dataset file (JSON/YAML)", file_types=[".json", ".yaml", ".yml"]
                    )
                    r_text = gr.Textbox(label="…or paste dataset (JSON/YAML)", lines=6)
                r_prompt = gr.Textbox(label="Prompt (system prompt)", lines=4)
                r_model = gr.Textbox(label="Model", value=DEFAULT_CANDIDATE_MODEL)
                r_run = gr.Button("Run", variant="primary")
                r_summary = gr.Markdown()
                r_df = gr.Dataframe(
                    headers=["#", "input", "output", "code", "judge", "human"],
                    interactive=False,
                    label="Per-test-case",
                )
                r_download = gr.File(label="Download report JSON")
                r_run.click(
                    run_fn,
                    inputs=[r_file, r_text, r_prompt, r_model],
                    outputs=[r_summary, r_df, r_download],
                )

    return app


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, **kwargs) -> None:
    """Build and launch the Gradio app."""
    build_app().launch(server_name=server_name, server_port=server_port, **kwargs)


if __name__ == "__main__":
    launch()
