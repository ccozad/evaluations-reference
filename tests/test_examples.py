"""Validate the shipped example fixtures load and are internally consistent."""

from pathlib import Path

import pytest

from evaluations_reference.lenses import code_checks
from evaluations_reference.models import EvalDataset, Lens

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_DIRS = ["json_extraction", "sentiment", "summarization"]


@pytest.mark.parametrize("name", EXAMPLE_DIRS)
def test_example_dataset_loads_and_is_consistent(name: str) -> None:
    folder = EXAMPLES / name
    dataset = EvalDataset.load(folder / "dataset.json")

    assert dataset.test_cases, f"{name} has no test cases"
    # Every configured code check must resolve to a registered grader.
    registered = set(code_checks.available_checks())
    for check in dataset.lens_config.checks:
        assert check.name in registered, f"{name}: unknown check {check.name!r}"
    # The judge lens needs a rubric to be meaningful.
    if Lens.JUDGE in dataset.lens_config.enabled:
        assert dataset.rubric.strip(), f"{name}: judge lens enabled but no rubric"


@pytest.mark.parametrize("name", EXAMPLE_DIRS)
def test_example_ships_two_prompt_variants(name: str) -> None:
    folder = EXAMPLES / name
    for variant in ("v1.txt", "v2.txt"):
        path = folder / variant
        assert path.exists(), f"{name} missing {variant}"
        assert path.read_text(encoding="utf-8").strip(), f"{name}/{variant} is empty"
