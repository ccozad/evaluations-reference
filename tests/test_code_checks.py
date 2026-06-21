"""Unit coverage for each built-in code-based check."""

import pytest

from evaluations_reference.lenses import code_checks
from evaluations_reference.models import TestCase

CASE = TestCase(input="q", expected="a")


def _run(name: str, output: str, **params: object) -> tuple[float, bool]:
    result = code_checks.run_check(name, CASE, output, dict(params))
    return result.score, result.passed


def test_length_bounds() -> None:
    assert _run("length_bounds", "hello", min=1, max=10) == (1.0, True)
    assert _run("length_bounds", "hello", min=10) == (0.0, False)
    assert _run("length_bounds", "hello world", max=5) == (0.0, False)


def test_keyword_present() -> None:
    assert _run("keyword_present", "the cat sat", keywords=["cat", "sat"]) == (1.0, True)
    assert _run("keyword_present", "the cat sat", keywords=["dog"]) == (0.0, False)
    assert _run("keyword_present", "The CAT", keywords=["cat"]) == (1.0, True)
    assert _run("keyword_present", "The CAT", keywords=["cat"], case_sensitive=True) == (0.0, False)


def test_keyword_absent() -> None:
    assert _run("keyword_absent", "all good", keywords=["bad"]) == (1.0, True)
    assert _run("keyword_absent", "this is bad", keywords=["bad"]) == (0.0, False)


def test_regex_match() -> None:
    assert _run("regex_match", "order #123", pattern=r"#\d+") == (1.0, True)
    assert _run("regex_match", "no number", pattern=r"#\d+") == (0.0, False)
    assert _run("regex_match", "HELLO", pattern="hello", ignore_case=True) == (1.0, True)


def test_json_valid() -> None:
    assert _run("json_valid", '{"a": 1}') == (1.0, True)
    assert _run("json_valid", "{not json}") == (0.0, False)


def test_python_valid() -> None:
    assert _run("python_valid", "def f():\n    return 1") == (1.0, True)
    assert _run("python_valid", "def f(:") == (0.0, False)


def test_regex_valid() -> None:
    assert _run("regex_valid", r"\d+[a-z]") == (1.0, True)
    assert _run("regex_valid", "[unclosed") == (0.0, False)


def test_max_sentences() -> None:
    assert _run("max_sentences", "One sentence.", max=1) == (1.0, True)
    assert _run("max_sentences", "One. Two. Three.", max=2) == (0.0, False)


def test_unknown_check_raises() -> None:
    with pytest.raises(KeyError):
        code_checks.run_check("does_not_exist", CASE, "out", {})


def test_registry_lists_all_builtins() -> None:
    names = set(code_checks.available_checks())
    assert {
        "length_bounds",
        "keyword_present",
        "keyword_absent",
        "regex_match",
        "json_valid",
        "python_valid",
        "regex_valid",
        "max_sentences",
    } <= names
