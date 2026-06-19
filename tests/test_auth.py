from __future__ import annotations

from src.api.auth import _is_authorized


def test_api_key_auth_is_optional_when_not_configured():
    assert _is_authorized(None, expected_api_key="")


def test_api_key_auth_requires_matching_key_when_configured():
    assert _is_authorized("secret", expected_api_key="secret")
    assert not _is_authorized(None, expected_api_key="secret")
    assert not _is_authorized("wrong", expected_api_key="secret")
