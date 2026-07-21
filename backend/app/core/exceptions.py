"""Shared error response helpers."""

from typing import Any


def unwrap_http_detail(detail: Any) -> dict:
    if isinstance(detail, dict):
        return detail
    return {"error": "HTTP_ERROR", "message": str(detail)}
