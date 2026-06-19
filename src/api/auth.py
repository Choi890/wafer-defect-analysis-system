from __future__ import annotations

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import API_KEY


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def _is_authorized(provided_api_key: str | None, expected_api_key: str = API_KEY) -> bool:
    if not expected_api_key:
        return True
    return bool(provided_api_key) and hmac.compare_digest(provided_api_key, expected_api_key)


async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not _is_authorized(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
