# Tutorial

A complete walkthrough of the eval harness: install, dataset format, the three
lenses, every CLI command, and the browser UI. For the elevator pitch and the
"what I'd change for production" notes, see the [README](README.md).

## Install

```sh
uv sync
```

Set `ANTHROPIC_API_KEY` for anything that calls a model (the judge lens, the
`compare`/`bootstrap` commands, the UI). A local `.env` works — the CLI loads it
automatically:

```sh
cp .env.example .env   # then edit it
```

Run the test suite with `uv run pytest`.

## Dataset format

A dataset is JSON or YAML. The shapes are Pydantic models, so malformed files
fail fast with a clear error.

```jsonc
{
  "name": "sentiment-classification",
  "description": "Classify a short text as positive, negative, or neutral.",
  "rubric": "Score relevance by whether the label is correct; faithfulness by ...",
  "lens_config": {
    "enabled": ["code", "judge"],            // which lenses run
    "checks": [                               // code-lens checks (in order)
      { "name": "contains_expected", "params": {} },
      { "name": "max_sentences", "params": { "max": 1 } }
    ]
  },
  "test_cases": [
    { "input": "I love this!", "expected": "positive", "metadata": { "label": "positive" } }
  ]
}
```

- **`input`** — the prompt or query under test.
- **`expected`** — an optional reference answer. Used by the `contains_expected`
  check and shown to the judge.
- **`metadata`** — arbitrary tags for slicing results.
- **`rubric`** — instructions for the model-as-judge.
- **`lens_config.enabled`** — any of `code`, `judge`, `human`.
- **`lens_config.checks`** — the code-lens checks; a case's code score is the
  mean of these.

Datasets are roundtrip-safe: load → save → load yields identical data, and JSON
and YAML are interchangeable.

## The three lenses

### Code lens

Deterministic checks from a registry. Each returns a 0–1 score; the case's code
score is their mean. Built-in checks:

| Check | Params | Passes when |
|---|---|---|
| `length_bounds` | `min`, `max` | output length (chars) is within bounds |
| `keyword_present` | `keywords`, `case_sensitive` | every keyword appears |
| `keyword_absent` | `keywords`, `case_sensitive` | no keyword appears |
| `regex_match` | `pattern`, `ignore_case` | the pattern matches (`re.search`) |
| `json_valid` | — | output parses as JSON |
| `python_valid` | — | output parses as Python (`ast.parse`) |
| `regex_valid` | — | output compiles as a regex |
| `max_sentences` | `max` | at most N sentences (split on `.!?`) |
| `contains_expected` | `case_sensitive` | the case's `expected` value appears in the output |

Add your own with the `@register("name")` decorator in
`src/evaluations_reference/lenses/code_checks.py`.

### Model-as-judge lens

An LLM scores each output for relevance (0–10) and faithfulness (0–10) against
the dataset rubric, plus a one-line reason; the two are normalized to a single
0–1 lens score. The default judge is `claude-sonnet-4-6` via the Anthropic SDK
using structured outputs. The model sits behind a `ModelJudge` protocol, so you
can swap in any backend without touching the runner. Enabling this lens without
`ANTHROPIC_API_KEY` set is a friendly, immediate error.

### Human lens

Prompts for a 1–5 rating per case in the terminal. It auto-skips (recording
`null` with a note) when stdin is not a TTY or when `SKIP_HUMAN_EVAL=1`, so it
never blocks CI or scripted runs.

## CLI commands

### `run`

```sh
uv run evals run examples/sentiment/dataset.json
```

Grades the dataset through its enabled lenses, prints a per-lens table, and
writes `<dataset>.report.json`. Human-readable output goes to stderr; stdout
gets the report path (so you can pipe it).

### `compare`

```sh
uv run evals compare examples/sentiment/dataset.json \
  --prompt-a examples/sentiment/v1.txt \
  --prompt-b examples/sentiment/v2.txt \
  [--model claude-sonnet-4-6] [--output cmp.json]
```

Runs both prompt variants against the dataset. A prompt file is used as the
**system prompt** and each test case's `input` as the user message. The two
variants' outputs are generated concurrently, graded by the enabled lenses, and
diffed: per-lens deltas, per-case winners, and an overall verdict.

### `bootstrap`

```sh
uv run evals bootstrap \
  --description "JSON extraction tests for invoices" \
  --count 10 --output invoices.json \
  [--rubric "Score faithfulness by ..."] [--model claude-sonnet-4-6]
```

Generates `N` test cases from a description using structured outputs, validates
each against the schema, and writes a dataset. Supplying `--rubric` enables the
judge lens on the result; otherwise the code lens is enabled.

### `describe`

```sh
uv run evals describe examples/sentiment/dataset.json
```

Prints the dataset's case count, enabled lenses, configured checks, and rubric —
no model calls.

### `ui`

```sh
uv run evals ui [--host 127.0.0.1] [--port 7860] [--share]
```

Launches the Gradio browser app (also runnable as
`python -m evaluations_reference.ui`).

## The browser UI

Three tabs over the same library functions:

1. **Compare prompts** — upload or paste a dataset, enter two prompts, run, and
   read aggregate deltas plus a per-test-case table. Click a row to expand the
   three lens verdicts for both prompts. Download the JSON report.
2. **Bootstrap dataset** — describe a dataset, pick a case count, optionally add
   a rubric, generate, and download the result.
3. **Single run** — grade one prompt against a dataset.

Long operations stream progress per test case. Missing `ANTHROPIC_API_KEY`
surfaces a clear message rather than a crash.

## Using the library directly

```python
import asyncio
from evaluations_reference import EvalDataset, run_eval

dataset = EvalDataset.load("examples/sentiment/dataset.json")
report = asyncio.run(run_eval(dataset))
print(report.lens_averages)
```

`run_eval`, `run_comparison`, and `bootstrap_dataset` are the library entry
points; each accepts injectable backends (judge, candidate, generator) so you
can test without network access.
