"""Bootstrapping: dataset assembly, roundtrip fidelity, and lens/rubric behavior."""

import asyncio
from pathlib import Path

import pytest

from evaluations_reference.bootstrap import GeneratedCase, bootstrap_dataset
from evaluations_reference.errors import DatasetValidationError
from evaluations_reference.models import EvalDataset, Lens


class FakeGenerator:
    """Returns a fixed list of generated cases — no network."""

    def __init__(self, n: int) -> None:
        self.cases = [GeneratedCase(input=f"input {i}", expected=f"expected {i}") for i in range(n)]

    async def generate(self, description: str, count: int) -> list[GeneratedCase]:
        return self.cases


def test_bootstrap_builds_dataset() -> None:
    ds = asyncio.run(bootstrap_dataset("invoice extraction", 3, FakeGenerator(3)))
    assert len(ds.test_cases) == 3
    assert ds.test_cases[0].input == "input 0"
    assert ds.lens_config.enabled == [Lens.CODE]
    assert ds.name == "invoice-extraction"


def test_bootstrap_with_rubric_enables_judge() -> None:
    ds = asyncio.run(bootstrap_dataset("q and a", 2, FakeGenerator(2), rubric="score relevance"))
    assert ds.lens_config.enabled == [Lens.JUDGE]
    assert ds.rubric == "score relevance"


def test_bootstrap_caps_at_requested_count() -> None:
    # Generator returns more than requested; bootstrap trims to count.
    ds = asyncio.run(bootstrap_dataset("topic", 2, FakeGenerator(5)))
    assert len(ds.test_cases) == 2


def test_bootstrap_invalid_count_raises() -> None:
    with pytest.raises(DatasetValidationError):
        asyncio.run(bootstrap_dataset("topic", 0, FakeGenerator(3)))


def test_bootstrapped_dataset_roundtrips(tmp_path: Path) -> None:
    ds = asyncio.run(bootstrap_dataset("topic", 4, FakeGenerator(4), rubric="r"))
    out = tmp_path / "boot.json"
    ds.save(out)
    assert EvalDataset.load(out) == ds
