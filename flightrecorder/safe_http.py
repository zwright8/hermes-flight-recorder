"""Dependency-free bounded HTTP policy for credential-bearing requests."""

from __future__ import annotations

import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO

from .redaction import is_secret_key

DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ERROR_BYTES = 4096
DEFAULT_MAX_SSE_LINE_BYTES = 64 * 1024
DEFAULT_MAX_SSE_EVENT_BYTES = 256 * 1024
DEFAULT_MAX_SSE_EVENTS = 128
DEFAULT_MAX_SSE_TOTAL_BYTES = 2 * 1024 * 1024


class SafeHttpError(ValueError):
    """Raised when a request violates the bounded HTTP policy or cannot complete."""


class HttpStatusError(SafeHttpError):
    """A non-success response with a bounded error body."""

    def __init__(self, status_code: int, body: bytes, *, truncated: bool) -> None:
        self.status_code = status_code
        self.body = body
        self.truncated = truncated
        suffix = " [truncated]" if truncated else ""
        super().__init__(f"HTTP status {status_code}: {body.decode('utf-8', errors='replace')}{suffix}")


@dataclass(frozen=True)
class SseReadResult:
    """Bounded Server-Sent Events data extracted from one response."""

    data_events: tuple[str, ...]
    done_seen: bool
    bytes_read: int


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject unsafe redirects before urllib constructs a redirected request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target_url = urllib.parse.urljoin(req.full_url, newurl)
        validate_http_url(target_url)
        source_origin = _http_origin(req.full_url)
        target_origin = _http_origin(target_url)
        if source_origin[0] == "https" and target_origin[0] == "http":
            raise SafeHttpError(
                "Blocked HTTPS downgrade redirect "
                f"from {_display_url(req.full_url)!r} to {_display_url(target_url)!r}"
            )
        if source_origin != target_origin and _request_has_credentials(req):
            raise SafeHttpError(
                "Blocked credentialed cross-origin redirect "
                f"from {_display_url(req.full_url)!r} to {_display_url(target_url)!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, target_url)


def bounded_http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    max_error_bytes: int = DEFAULT_MAX_ERROR_BYTES,
) -> tuple[int, bytes]:
    """Perform one request and return a strictly bounded success body."""
    with open_http_response(
        method,
        url,
        headers=headers,
        data=data,
        timeout=timeout,
        max_error_bytes=max_error_bytes,
    ) as response:
        body = read_bounded_bytes(
            response,
            max_bytes=max_body_bytes,
            label=f"HTTP {method.upper()} response body",
        )
        return int(response.status), body


def open_http_response(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float,
    max_error_bytes: int = DEFAULT_MAX_ERROR_BYTES,
) -> Any:
    """Open a validated request with redirect and bounded-error policy applied."""
    validate_http_url(url)
    _validate_limit(max_error_bytes, "max_error_bytes", allow_zero=True)
    try:
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers or {},
            method=method.upper(),
        )
        opener = urllib.request.build_opener(SafeRedirectHandler())
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            try:
                error_body, truncated = _read_clipped_bytes(exc, max_bytes=max_error_bytes)
            except (TimeoutError, OSError, http.client.HTTPException) as body_exc:
                raise SafeHttpError(
                    f"HTTP status {int(exc.code)} error body read failed: {body_exc}"
                ) from body_exc
        finally:
            exc.close()
        raise HttpStatusError(int(exc.code), error_body, truncated=truncated) from exc
    except SafeHttpError:
        raise
    except urllib.error.URLError as exc:
        raise SafeHttpError(f"HTTP request failed: {exc.reason}") from exc
    except (TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
        raise SafeHttpError(f"HTTP request failed: {exc}") from exc


def read_bounded_bytes(stream: BinaryIO, *, max_bytes: int, label: str) -> bytes:
    """Read at most ``max_bytes`` and fail if the stream contains more data."""
    _validate_limit(max_bytes, "max_bytes", allow_zero=True)
    try:
        payload = stream.read(max_bytes + 1)
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
        raise SafeHttpError(f"{label} read failed: {exc}") from exc
    if len(payload) > max_bytes:
        raise SafeHttpError(f"{label} exceeded max_bytes={max_bytes}")
    return payload


def read_bounded_sse(
    stream: BinaryIO,
    *,
    max_line_bytes: int = DEFAULT_MAX_SSE_LINE_BYTES,
    max_event_bytes: int = DEFAULT_MAX_SSE_EVENT_BYTES,
    max_events: int = DEFAULT_MAX_SSE_EVENTS,
    max_total_bytes: int = DEFAULT_MAX_SSE_TOTAL_BYTES,
    done_value: str = "[DONE]",
) -> SseReadResult:
    """Read SSE data with independent line, event, count, and aggregate bounds."""
    for value, label in (
        (max_line_bytes, "max_line_bytes"),
        (max_event_bytes, "max_event_bytes"),
        (max_events, "max_events"),
        (max_total_bytes, "max_total_bytes"),
    ):
        _validate_limit(value, label, allow_zero=False)

    events: list[str] = []
    data_lines: list[str] = []
    event_bytes = 0
    total_bytes = 0
    done_seen = False

    def finish_event() -> bool:
        nonlocal event_bytes, done_seen
        if not data_lines:
            event_bytes = 0
            return False
        payload = "\n".join(data_lines)
        data_lines.clear()
        event_bytes = 0
        if payload.strip() == done_value:
            done_seen = True
            return True
        if len(events) >= max_events:
            raise SafeHttpError(f"SSE event count exceeded max_events={max_events}")
        events.append(payload)
        return False

    while True:
        try:
            raw_line = stream.readline(max_line_bytes + 1)
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            raise SafeHttpError(f"SSE read failed: {exc}") from exc
        if not raw_line:
            finish_event()
            break
        total_bytes += len(raw_line)
        if len(raw_line) > max_line_bytes:
            raise SafeHttpError(f"SSE line exceeded max_line_bytes={max_line_bytes}")
        if total_bytes > max_total_bytes:
            raise SafeHttpError(f"SSE aggregate exceeded max_total_bytes={max_total_bytes}")

        event_bytes += len(raw_line)
        if event_bytes > max_event_bytes:
            raise SafeHttpError(f"SSE event exceeded max_event_bytes={max_event_bytes}")

        line = raw_line.rstrip(b"\r\n")
        if not line:
            if finish_event():
                break
            continue
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value.decode("utf-8", errors="replace"))

    return SseReadResult(tuple(events), done_seen, total_bytes)


def validate_http_url(url: str) -> None:
    """Require an absolute HTTP(S) URL with a valid hostname and port."""
    try:
        parsed = urllib.parse.urlsplit(url)
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise SafeHttpError(f"HTTP URL is invalid: {_display_url(str(url))!r}") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise SafeHttpError(f"HTTP URL must use http or https: {_display_url(url)!r}")


def _read_clipped_bytes(stream: BinaryIO, *, max_bytes: int) -> tuple[bytes, bool]:
    payload = stream.read(max_bytes + 1)
    return payload[:max_bytes], len(payload) > max_bytes


def _validate_limit(value: int, label: str, *, allow_zero: bool) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SafeHttpError(f"{label} must be a {qualifier} integer")


def _http_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parsed.hostname or "").lower(), port


def _request_has_credentials(request: urllib.request.Request) -> bool:
    parsed = urllib.parse.urlsplit(request.full_url)
    if parsed.username is not None or parsed.password is not None:
        return True
    return any(is_secret_key(name) for name, _value in request.header_items())


def _display_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
    except ValueError:
        return "<invalid-url>"
