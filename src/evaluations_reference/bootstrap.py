"""Dataset bootstrapping: generate test cases from a natural-language description.

A structured-output call asks Claude for ``N`` diverse test cases; each is
validated against the dataset schema before being assembled into a saveable
:class:`EvalDataset`.
"""

from __future__ import annotations

import re
from typing import Protocol

import anthropic
from pydantic import BaseModel

from evaluations_reference.client import require_api_key, translate_api_errors
from evaluations_reference.errors import DatasetValidationError, UpstreamAPIError
from evaluations_reference.models import EvalDataset, Lens, LensConfig, TestCase

# Default model for generating test cases; override via the CLI --model flag.
DEFAULT_BOOTSTRAP_MODEL = "claude-sonnet-4-6"


class GeneratedCase(BaseModel):
    """A single model-generated test case (kept flat for structured outputs)."""

    input: str
    expected: str


class GeneratedCases(BaseModel):
    """The structured-output contract: a list of generated cases."""

    cases: list[GeneratedCase]


class DatasetGenerator(Protocol):
    """A swappable test-case generator backend."""

    async def generate(self, description: str, count: int) -> list[GeneratedCase]: ...


_SYSTEM_PROMPT = (
    "You generate evaluation test cases for LLM systems. Given a natural-language "
    "description of what to test, produce diverse, realistic cases. Each case has an "
    "`input` (the prompt or query under test) and an `expected` (a concise reference "
    "answer). Vary topic, phrasing, and difficulty across the cases; avoid duplicates."
)


def _build_prompt(description: str, count: int) -> str:
    return (
        f"Generate exactly {count} diverse test cases for the following dataset.\n\n"
        f"Description: {description}\n\n"
        "Return them via the structured output schema."
    )


class AnthropicGenerator:
    """Default generator backed by the Anthropic API and structured outputs."""

    def __init__(self, model: str = DEFAULT_BOOTSTRAP_MODEL, max_tokens: int = 4096) -> None:
        require_api_key("Dataset bootstrapping")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic()

    async def generate(self, description: str, count: int) -> list[GeneratedCase]:
        with translate_api_errors("bootstrap model call"):
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_prompt(description, count)}],
                output_format=GeneratedCases,
            )
        parsed = response.parsed_output
        if parsed is None:
            raise UpstreamAPIError("bootstrap model returned no parseable output")
        return parsed.cases


def _slug(description: str, max_words: int = 5) -> str:
    words = re.findall(r"[a-z0-9]+", description.lower())
    slug = "-".join(words[:max_words])
    return slug or "bootstrapped-dataset"


async def bootstrap_dataset(
    description: str,
    count: int,
    generator: DatasetGenerator,
    name: str | None = None,
    rubric: str | None = None,
) -> EvalDataset:
    """Generate a dataset of ``count`` cases from ``description``.

    Enables the judge lens when a ``rubric`` is supplied, otherwise the code
    lens. Raises :class:`DatasetValidationError` if ``count`` is not positive.
    """
    if count < 1:
        raise DatasetValidationError(f"count must be >= 1, got {count}")

    generated = await generator.generate(description, count)
    # Validate each case against the schema and cap at the requested count.
    test_cases = [TestCase(input=c.input, expected=c.expected) for c in generated[:count]]

    enabled = [Lens.JUDGE] if rubric else [Lens.CODE]
    return EvalDataset(
        name=name or _slug(description),
        description=description,
        rubric=rubric or "",
        lens_config=LensConfig(enabled=enabled),
        test_cases=test_cases,
    )
