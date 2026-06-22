"""Shared helpers for the Anthropic-backed components (judge, candidate, generator).

Keeps API-key checking and error translation in one place so each backend maps
``anthropic`` exceptions to the framework's typed errors identically.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import anthropic

from evaluations_reference.errors import (
    MissingAPIKeyError,
    UpstreamAPIError,
    UpstreamTimeoutError,
)


def require_api_key(what: str) -> None:
    """Raise :class:`MissingAPIKeyError` if ``ANTHROPIC_API_KEY`` is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            f"{what} requires ANTHROPIC_API_KEY, which is not set. "
            "Set the key (e.g. in a local .env) to continue."
        )


@contextlib.contextmanager
def translate_api_errors(what: str) -> Iterator[None]:
    """Translate ``anthropic`` exceptions into the framework's typed errors.

    ``what`` names the operation for the error message (e.g. "judge model call").
    """
    try:
        yield
    except anthropic.APITimeoutError as exc:
        raise UpstreamTimeoutError(f"{what} timed out: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise UpstreamAPIError(
            f"{what} failed: {exc.message}", status_code=exc.status_code
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise UpstreamAPIError(f"{what} connection error: {exc}") from exc
