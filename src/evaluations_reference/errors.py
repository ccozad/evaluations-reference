"""Typed errors for the eval framework.

The runner and CLI raise these so callers can distinguish a malformed dataset
(caught before any LLM call) from an upstream API failure or timeout.
"""

from __future__ import annotations


class EvalError(Exception):
    """Base class for all eval-framework errors."""


class DatasetValidationError(EvalError):
    """A dataset or test case is malformed. Raised before any LLM call."""


class MissingAPIKeyError(EvalError):
    """The judge lens is enabled but ``ANTHROPIC_API_KEY`` is not set."""


class UpstreamAPIError(EvalError):
    """An Anthropic API call returned an error. Preserves the HTTP status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UpstreamTimeoutError(EvalError):
    """An Anthropic API call timed out."""


class EmptyDatasetWarning(EvalError):
    """A dataset has no test cases. The CLI treats this as exit 0 with a warning."""
