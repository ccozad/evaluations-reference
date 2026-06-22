"""Pydantic data models for the eval framework: datasets, configuration, and reports.

Datasets load from and save to JSON or YAML, and are roundtrip-safe — loading a
file, saving it, and loading it again produces identical data.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Lens(StrEnum):
    """The three complementary evaluation lenses."""

    CODE = "code"
    JUDGE = "judge"
    HUMAN = "human"


class CheckSpec(BaseModel):
    """One enabled code-based check and its parameters.

    `name` resolves to a grader in the check registry; `params` is passed
    through to that grader (e.g. ``{"min": 1, "max": 200}`` for length bounds).
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class LensConfig(BaseModel):
    """Which lenses are enabled for a dataset, plus per-lens settings."""

    enabled: list[Lens] = Field(default_factory=lambda: [Lens.CODE])
    checks: list[CheckSpec] = Field(default_factory=list)


class TestCase(BaseModel):
    """A single evaluation example.

    `input` is the prompt/input under test, `expected` an optional reference
    answer, and `metadata` arbitrary tags used for slicing results.
    """

    # Tell pytest this is not a test class (name starts with "Test").
    __test__ = False

    input: str
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A named collection of test cases plus lens configuration and rubric."""

    name: str
    description: str = ""
    rubric: str = ""
    lens_config: LensConfig = Field(default_factory=LensConfig)
    test_cases: list[TestCase] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        """Load a dataset from a ``.json`` or ``.yaml``/``.yml`` file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        """Save a dataset to a ``.json`` or ``.yaml``/``.yml`` file."""
        path = Path(path)
        data = self.model_dump(mode="json")
        if path.suffix.lower() in {".yaml", ".yml"}:
            text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")


class CheckResult(BaseModel):
    """The outcome of one code-based check against a test case."""

    name: str
    score: float
    passed: bool
    reason: str


class LensResult(BaseModel):
    """A single lens's outcome for one test case.

    `score` is normalized to 0-1 for aggregation, or ``None`` when the lens was
    skipped (e.g. human-grade outside a TTY). `detail` carries lens-specific
    structure (per-check results, judge sub-scores, etc.).
    """

    lens: Lens
    score: float | None
    note: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class TestCaseResult(BaseModel):
    """All lens results for a single test case, plus the produced output."""

    # Tell pytest this is not a test class (name starts with "Test").
    __test__ = False

    test_case: TestCase
    output: str
    lens_results: list[LensResult] = Field(default_factory=list)


class EvalReport(BaseModel):
    """The aggregated result of running a dataset through its enabled lenses."""

    dataset_name: str
    case_count: int
    lens_averages: dict[Lens, float | None] = Field(default_factory=dict)
    results: list[TestCaseResult] = Field(default_factory=list)
