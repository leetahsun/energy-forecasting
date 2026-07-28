"""Retry-with-backoff wrapper for external HTTP calls.
"""

import time

import requests


def get_with_retry(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code >= 500:
                last_exception = requests.exceptions.HTTPError(
                    f"{resp.status_code} server error", response=resp
                )
            else:
                resp.raise_for_status()  # raises on 4xx, not retried (see below)
                return resp
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise  # client errors are not transient -- fail immediately
            last_exception = e
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e

        if attempt < max_retries - 1:
            time.sleep(backoff_seconds * (2 ** attempt))

    raise last_exception