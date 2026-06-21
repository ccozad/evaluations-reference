"""Lens 1: code-based grading.

An extensible registry maps a check name to a grader function. Each grader takes
the test case and the produced output and returns ``(score, passed, reason)``
where ``score`` is in 0-1. A test case's code-based score is the mean of its
enabled checks.

Register a new check with the ``@register("name")`` decorator; it then becomes
usable from a dataset's ``lens_config.checks`` by name.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from typing import Any

from evaluations_reference.models import CheckResult, TestCase

# A grader returns (score_0_to_1, passed, reason).
Grader = Callable[[TestCase, str, dict[str, Any]], "tuple[float, bool, str]"]

_REGISTRY: dict[str, Grader] = {}


def register(name: str) -> Callable[[Grader], Grader]:
    """Register a grader under ``name`` in the check registry."""

    def decorator(fn: Grader) -> Grader:
        if name in _REGISTRY:
            raise ValueError(f"check {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_checks() -> list[str]:
    """Return the names of all registered checks, sorted."""
    return sorted(_REGISTRY)


def run_check(name: str, case: TestCase, output: str, params: dict[str, Any]) -> CheckResult:
    """Run a single registered check, returning a structured result."""
    grader = _REGISTRY.get(name)
    if grader is None:
        raise KeyError(f"unknown check {name!r}; available: {', '.join(available_checks())}")
    score, passed, reason = grader(case, output, params)
    return CheckResult(name=name, score=score, passed=passed, reason=reason)


def _bool_result(passed: bool, reason: str) -> tuple[float, bool, str]:
    return (1.0 if passed else 0.0, passed, reason)


@register("length_bounds")
def _length_bounds(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when ``len(output)`` is within ``[min, max]`` characters."""
    low = params.get("min", 0)
    high = params.get("max")
    n = len(output)
    if n < low:
        return _bool_result(False, f"length {n} < min {low}")
    if high is not None and n > high:
        return _bool_result(False, f"length {n} > max {high}")
    return _bool_result(True, f"length {n} within bounds")


def _normalize(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


@register("keyword_present")
def _keyword_present(
    case: TestCase, output: str, params: dict[str, Any]
) -> tuple[float, bool, str]:
    """Pass when every keyword in ``keywords`` appears in the output."""
    keywords: list[str] = params.get("keywords", [])
    cs = bool(params.get("case_sensitive"))
    haystack = _normalize(output, cs)
    missing = [kw for kw in keywords if _normalize(kw, cs) not in haystack]
    if missing:
        return _bool_result(False, f"missing keywords: {', '.join(missing)}")
    return _bool_result(True, "all keywords present")


@register("keyword_absent")
def _keyword_absent(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when no keyword in ``keywords`` appears in the output."""
    keywords: list[str] = params.get("keywords", [])
    cs = bool(params.get("case_sensitive"))
    haystack = _normalize(output, cs)
    present = [kw for kw in keywords if _normalize(kw, cs) in haystack]
    if present:
        return _bool_result(False, f"forbidden keywords present: {', '.join(present)}")
    return _bool_result(True, "no forbidden keywords")


@register("regex_match")
def _regex_match(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when the output matches the ``pattern`` regex (``re.search``)."""
    pattern = params.get("pattern", "")
    flags = re.IGNORECASE if params.get("ignore_case") else 0
    if re.search(pattern, output, flags):
        return _bool_result(True, f"matched /{pattern}/")
    return _bool_result(False, f"did not match /{pattern}/")


@register("json_valid")
def _json_valid(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when the output parses as JSON."""
    try:
        json.loads(output)
    except ValueError as exc:
        return _bool_result(False, f"invalid JSON: {exc}")
    return _bool_result(True, "valid JSON")


@register("python_valid")
def _python_valid(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when the output parses as Python source (``ast.parse``)."""
    try:
        ast.parse(output)
    except SyntaxError as exc:
        return _bool_result(False, f"invalid Python: {exc}")
    return _bool_result(True, "valid Python syntax")


@register("regex_valid")
def _regex_valid(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when the output compiles as a regular expression."""
    try:
        re.compile(output)
    except re.error as exc:
        return _bool_result(False, f"invalid regex: {exc}")
    return _bool_result(True, "valid regex")


@register("max_sentences")
def _max_sentences(case: TestCase, output: str, params: dict[str, Any]) -> tuple[float, bool, str]:
    """Pass when the output has at most ``max`` sentences (split on . ! ?)."""
    limit = params.get("max", 1)
    sentences = [s for s in re.split(r"[.!?]+", output) if s.strip()]
    n = len(sentences)
    if n > limit:
        return _bool_result(False, f"{n} sentences > max {limit}")
    return _bool_result(True, f"{n} sentences within max {limit}")
