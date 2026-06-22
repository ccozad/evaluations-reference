"""Human-grade lens: skip detection and the interactive rating prompt."""

import builtins
import sys

import pytest

from evaluations_reference.lenses import human
from evaluations_reference.models import TestCase

CASE = TestCase(input="q", expected="a")


def test_should_skip_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_HUMAN_EVAL", "1")
    skip, reason = human.should_skip()
    assert skip and "SKIP_HUMAN_EVAL" in reason


def test_should_skip_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKIP_HUMAN_EVAL", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    skip, reason = human.should_skip()
    assert skip and "TTY" in reason


def test_should_not_skip_in_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKIP_HUMAN_EVAL", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    skip, reason = human.should_skip()
    assert not skip and reason == ""


def test_prompt_rating_accepts_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "3")
    assert human.prompt_rating(CASE, "output") == 3


def test_prompt_rating_reprompts_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["0", "nine", "4"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    assert human.prompt_rating(CASE, "output") == 4
