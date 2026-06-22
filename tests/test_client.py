"""Coverage for the shared API helpers: key checking and error translation."""

import anthropic
import httpx
import pytest

from evaluations_reference.client import require_api_key, translate_api_errors
from evaluations_reference.errors import (
    MissingAPIKeyError,
    UpstreamAPIError,
    UpstreamTimeoutError,
)

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_require_api_key_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        require_api_key("X")


def test_require_api_key_ok_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    require_api_key("X")  # does not raise


def test_translate_timeout() -> None:
    with pytest.raises(UpstreamTimeoutError):
        with translate_api_errors("call"):
            raise anthropic.APITimeoutError(request=_REQ)


def test_translate_status_error_preserves_code() -> None:
    resp = httpx.Response(429, request=_REQ)
    with pytest.raises(UpstreamAPIError) as exc:
        with translate_api_errors("call"):
            raise anthropic.APIStatusError("rate limited", response=resp, body=None)
    assert exc.value.status_code == 429


def test_translate_connection_error() -> None:
    with pytest.raises(UpstreamAPIError):
        with translate_api_errors("call"):
            raise anthropic.APIConnectionError(message="boom", request=_REQ)
