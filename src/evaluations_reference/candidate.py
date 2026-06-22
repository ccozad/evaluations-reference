"""The system under test: a prompt variant that produces an output per test case.

A prompt variant is a system prompt; each test case's ``input`` is sent as the
user message and the model's text reply is the output the lenses grade. The
model is hidden behind the ``CandidateModel`` protocol so the comparison flow
never depends on a specific backend (and can be mocked in tests).
"""

from __future__ import annotations

from typing import Protocol

import anthropic

from evaluations_reference.client import require_api_key, translate_api_errors
from evaluations_reference.models import TestCase

# Default system-under-test model; override via the CLI --model flag.
DEFAULT_CANDIDATE_MODEL = "claude-sonnet-4-6"


class CandidateModel(Protocol):
    """A swappable system-under-test backend.

    Runs one test case through a prompt variant and returns the output text.
    """

    async def generate(self, prompt: str, case: TestCase) -> str: ...


class AnthropicCandidate:
    """Default candidate backed by the Anthropic API.

    ``prompt`` is used as the system prompt and ``case.input`` as the user
    message.
    """

    def __init__(self, model: str = DEFAULT_CANDIDATE_MODEL, max_tokens: int = 1024) -> None:
        require_api_key("Prompt comparison")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic()

    async def generate(self, prompt: str, case: TestCase) -> str:
        with translate_api_errors("candidate model call"):
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=prompt,
                messages=[{"role": "user", "content": case.input}],
            )
        return "".join(block.text for block in response.content if block.type == "text")
