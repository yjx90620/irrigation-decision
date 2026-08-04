"""Shared HTTP request-with-retry (audit-v3, 4.5).

The downloaders previously could exhaust their retry loops and then parse
the LAST FAILED response (e.g. a 429 body) as if it were data. This
helper makes "return a valid response or raise" the only exit path.
"""

import time


def request_with_retry(
    request_fn,
    *,
    max_attempts: int = 6,
    retry_statuses: set = {429, 500, 502, 503, 504},
    base_wait: float = 2.0,
):
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = request_fn()
        except Exception as exc:  # connection-level failures
            last_error = RuntimeError(f"request raised {type(exc).__name__}: {exc}")
            time.sleep(base_wait * (attempt + 1))
            continue
        if response.ok:
            return response
        if response.status_code not in retry_statuses:
            response.raise_for_status()
        last_error = RuntimeError(
            f"HTTP {response.status_code}: {response.text[:500]}"
        )
        time.sleep(min(base_wait * (2 ** attempt), 60))
    raise RuntimeError(
        f"Request failed after {max_attempts} attempts"
    ) from last_error
