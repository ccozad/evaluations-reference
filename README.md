# Evaluations Reference

A production-mindful, three-lens evaluation harness for LLM systems. It combines
deterministic code checks, model-as-judge, and human-grade review with a
side-by-side prompt comparison flow that answers the only question that matters
when you change a prompt: **did this actually make things better?**

> **Status:** eval framework, prompt comparison, and browser UI landed; docs polish in M4.

## Browser UI

```sh
evals ui    # then open http://127.0.0.1:7860
```

![Evaluations Reference UI](docs/ui-screenshot.png)

The side-by-side comparison tab after a run — per-lens score deltas, per-test-case
winners, and an overall verdict. Two more tabs cover dataset bootstrapping and
single-prompt eval runs.

## Install

```sh
uv sync
```

## Run tests

```sh
uv run pytest
```

Real usage instructions land in M4.

## License

[MIT](LICENSE)
