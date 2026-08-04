"""Request guards and response security headers for the public API."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from web.backend.settings import AppSettings

PREDICTION_PATHS = frozenset({"/predict", "/v1/predict"})
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


class RequestRateLimiter:
    """Small per-process fixed-window limiter for unauthenticated inference calls."""

    def __init__(self, limit_per_minute: int) -> None:
        self._limit_per_minute = limit_per_minute
        self._lock = threading.Lock()
        self._requests_by_client: dict[str, deque[float]] = {}

    def allow(self, client_id: str, now: float | None = None) -> bool:
        if self._limit_per_minute <= 0:
            return True

        current_time = now if now is not None else time.monotonic()
        window_start = current_time - 60
        with self._lock:
            timestamps = self._requests_by_client.setdefault(client_id, deque())
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            if len(timestamps) >= self._limit_per_minute:
                return False
            timestamps.append(current_time)
            return True


def _client_id(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


def _content_length(request: Request) -> int | None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        return None
    try:
        return int(raw_content_length)
    except ValueError:
        return -1


def _guard_prediction_request(
    request: Request,
    settings: AppSettings,
    rate_limiter: RequestRateLimiter,
) -> JSONResponse | None:
    content_length = _content_length(request)
    if content_length is None:
        return JSONResponse(
            status_code=411,
            content={"detail": "Content-Length is required for prediction requests."},
        )
    if content_length < 0 or content_length > settings.max_body_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body must be {settings.max_body_bytes} bytes or fewer."},
        )
    if not rate_limiter.allow(_client_id(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many prediction requests. Please wait and try again."},
        )
    return None


class PredictionSecurityMiddleware(BaseHTTPMiddleware):
    """Protect inference routes and attach browser-safe response headers."""

    def __init__(self, app, *, settings: AppSettings) -> None:
        super().__init__(app)
        self._settings = settings
        self._rate_limiter = RequestRateLimiter(settings.rate_limit_per_minute)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response: Response
        is_prediction = request.method.upper() == "POST" and request.url.path in PREDICTION_PATHS
        guard_response = (
            _guard_prediction_request(request, self._settings, self._rate_limiter)
            if is_prediction
            else None
        )
        response = guard_response if guard_response is not None else await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response
