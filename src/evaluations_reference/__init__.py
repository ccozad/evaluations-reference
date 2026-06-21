"""Evaluations Reference — a three-lens evaluation harness for LLM systems."""

from evaluations_reference.models import (
    CheckSpec,
    EvalDataset,
    EvalReport,
    Lens,
    LensConfig,
    TestCase,
)
from evaluations_reference.runner import run_eval

__version__ = "0.1.0"

__all__ = [
    "CheckSpec",
    "EvalDataset",
    "EvalReport",
    "Lens",
    "LensConfig",
    "TestCase",
    "run_eval",
    "__version__",
]
