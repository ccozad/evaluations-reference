"""Lens 2: model-as-judge.

An LLM scores each test case against the dataset's rubric. The model is hidden
behind the ``ModelJudge`` protocol so the runner never depends on a specific
backend; the default ``AnthropicJudge`` calls Claude Sonnet via the ``anthropic``
SDK and uses structured outputs so the scores come back already validated.
"""

from __future__ import annotations

import os
from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

from evaluations_reference.errors import (
    MissingAPIKeyError,
    UpstreamAPIError,
    UpstreamTimeoutError,
)
from evaluations_reference.models import TestCase

# Claude Sonnet 4.x — the dataset rubric default. Swappable via AnthropicJudge(model=...).
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


class JudgeScores(BaseModel):
    """Structured judge output. Relevance and faithfulness are each 0-10."""

    relevance: int = Field(ge=0, le=10)
    faithfulness: int = Field(ge=0, le=10)
    reasoning: str


class ModelJudge(Protocol):
    """A swappable judge backend.

    Implementations score one test case against a rubric and return raw 0-10
    sub-scores plus a short reasoning string.
    """

    async def score(self, case: TestCase, output: str, rubric: str) -> JudgeScores: ...


_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. Score the assistant output against the "
    "provided rubric. Return a relevance score (0-10) for how well the output "
    "addresses the input, a faithfulness score (0-10) for how well it is grounded "
    "in the input and any reference answer, and a one-sentence reasoning."
)


def _build_prompt(case: TestCase, output: str, rubric: str) -> str:
    parts = [f"Rubric:\n{rubric or '(no rubric provided; judge on general quality)'}"]
    parts.append(f"\nInput:\n{case.input}")
    if case.expected is not None:
        parts.append(f"\nReference answer:\n{case.expected}")
    parts.append(f"\nAssistant output:\n{output}")
    return "\n".join(parts)


class AnthropicJudge:
    """Default judge backed by the Anthropic API and structured outputs."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, max_tokens: int = 1024) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingAPIKeyError(
                "The judge lens is enabled but ANTHROPIC_API_KEY is not set. "
                "Set the key, or remove the judge lens from the dataset's lens_config."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic()

    async def score(self, case: TestCase, output: str, rubric: str) -> JudgeScores:
        try:
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_prompt(case, output, rubric)}],
                output_format=JudgeScores,
            )
        except anthropic.APITimeoutError as exc:
            raise UpstreamTimeoutError(f"judge model call timed out: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise UpstreamAPIError(
                f"judge model call failed: {exc.message}", status_code=exc.status_code
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise UpstreamAPIError(f"judge model connection error: {exc}") from exc

        parsed = response.parsed_output
        if parsed is None:
            raise UpstreamAPIError("judge model returned no parseable output")
        return parsed
