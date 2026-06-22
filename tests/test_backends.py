"""Cover the Anthropic-backed backends with a fake client (no network)."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from evaluations_reference.bootstrap import (
    AnthropicGenerator,
    GeneratedCase,
    GeneratedCases,
)
from evaluations_reference.candidate import AnthropicCandidate
from evaluations_reference.errors import UpstreamAPIError
from evaluations_reference.lenses.judge import AnthropicJudge, JudgeScores
from evaluations_reference.models import TestCase

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class _FakeMessages:
    def __init__(self, create_result=None, parse_result=None, raises=None) -> None:
        self._create_result = create_result
        self._parse_result = parse_result
        self._raises = raises

    async def create(self, **kwargs):
        if self._raises:
            raise self._raises
        return self._create_result

    async def parse(self, **kwargs):
        if self._raises:
            raise self._raises
        return self._parse_result


class _FakeClient:
    def __init__(self, **kwargs) -> None:
        self.messages = _FakeMessages(**kwargs)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")


def test_candidate_generate_joins_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    content = [
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="text", text="world"),
    ]
    cand = AnthropicCandidate()
    monkeypatch.setattr(
        cand, "_client", _FakeClient(create_result=SimpleNamespace(content=content))
    )
    out = asyncio.run(cand.generate("sys", TestCase(input="hi")))
    assert out == "Hello world"


def test_candidate_translates_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = httpx.Response(500, request=_REQ)
    cand = AnthropicCandidate()
    monkeypatch.setattr(
        cand,
        "_client",
        _FakeClient(raises=anthropic.APIStatusError("boom", response=resp, body=None)),
    )
    with pytest.raises(UpstreamAPIError):
        asyncio.run(cand.generate("sys", TestCase(input="hi")))


def test_judge_score_returns_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = JudgeScores(relevance=9, faithfulness=7, reasoning="ok")
    judge = AnthropicJudge()
    monkeypatch.setattr(
        judge, "_client", _FakeClient(parse_result=SimpleNamespace(parsed_output=parsed))
    )
    scores = asyncio.run(judge.score(TestCase(input="q"), "out", "rubric"))
    assert scores.relevance == 9 and scores.faithfulness == 7


def test_judge_raises_on_empty_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = AnthropicJudge()
    monkeypatch.setattr(
        judge, "_client", _FakeClient(parse_result=SimpleNamespace(parsed_output=None))
    )
    with pytest.raises(UpstreamAPIError):
        asyncio.run(judge.score(TestCase(input="q"), "out", "rubric"))


def test_generator_returns_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = GeneratedCases(cases=[GeneratedCase(input="i", expected="e")])
    gen = AnthropicGenerator()
    monkeypatch.setattr(
        gen, "_client", _FakeClient(parse_result=SimpleNamespace(parsed_output=cases))
    )
    result = asyncio.run(gen.generate("desc", 1))
    assert result[0].input == "i"


def test_generator_raises_on_empty_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = AnthropicGenerator()
    monkeypatch.setattr(
        gen, "_client", _FakeClient(parse_result=SimpleNamespace(parsed_output=None))
    )
    with pytest.raises(UpstreamAPIError):
        asyncio.run(gen.generate("desc", 1))
