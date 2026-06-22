"""Evaluations Reference — a three-lens evaluation harness for LLM systems."""

from evaluations_reference.bootstrap import bootstrap_dataset
from evaluations_reference.compare import ComparisonReport, run_comparison
from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    EvalReport,
    Lens,
    LensConfig,
    TestCase,
)
from evaluations_reference.runner import run_eval

__version__ = "0.3.0"

__all__ = [
    "CheckSpec",
    "ComparisonReport",
    "EvalDataset",
    "EvalReport",
    "Lens",
    "LensConfig",
    "TestCase",
    "bootstrap_dataset",
    "run_comparison",
    "run_eval",
    "__version__",
]
