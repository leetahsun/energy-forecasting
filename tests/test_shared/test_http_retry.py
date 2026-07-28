"""Tests for shared/http_retry.py."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.http_retry import get_with_retry


def test_succeeds_immediately_with_no_retry_needed():
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status.return_value = None

    with patch("shared.http_retry.requests.get", return_value=mock_resp) as mock_get:
        resp = get_with_retry("https://example.com", max_retries=3, backoff_seconds=0.01)

    assert resp is mock_resp
    assert mock_get.call_count == 1


def test_recovers_after_transient_timeout():
    """The exact scenario observed in production: a ReadTimeout on the
    first attempt, success on the next.
    """
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status.return_value = None

    with patch(
        "shared.http_retry.requests.get",
        side_effect=[requests.exceptions.ReadTimeout("timed out"), mock_resp],
    ) as mock_get:
        resp = get_with_retry("https://example.com", max_retries=3, backoff_seconds=0.01)

    assert resp is mock_resp
    assert mock_get.call_count == 2


def test_raises_after_exhausting_all_retries():
    with patch(
        "shared.http_retry.requests.get",
        side_effect=requests.exceptions.ReadTimeout("timed out"),
    ) as mock_get:
        with pytest.raises(requests.exceptions.ReadTimeout):
            get_with_retry("https://example.com", max_retries=3, backoff_seconds=0.01)

    assert mock_get.call_count == 3


def test_does_not_retry_on_4xx_client_error():
    """A 404 (e.g. a bad filter ID) won't succeed on retry -- should fail
    immediately, not waste time/requests retrying something that can't
    change.
    """
    mock_resp = MagicMock(status_code=404)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error", response=mock_resp
    )

    with patch("shared.http_retry.requests.get", return_value=mock_resp) as mock_get:
        with pytest.raises(requests.exceptions.HTTPError):
            get_with_retry("https://example.com", max_retries=3, backoff_seconds=0.01)

    assert mock_get.call_count == 1


def test_retries_on_5xx_server_error():
    mock_resp_500 = MagicMock(status_code=500)
    mock_resp_ok = MagicMock(status_code=200)
    mock_resp_ok.raise_for_status.return_value = None

    with patch(
        "shared.http_retry.requests.get", side_effect=[mock_resp_500, mock_resp_ok]
    ) as mock_get:
        resp = get_with_retry("https://example.com", max_retries=3, backoff_seconds=0.01)

    assert resp is mock_resp_ok
    assert mock_get.call_count == 2


def test_backoff_grows_exponentially():
    import shared.http_retry as http_retry_module

    sleep_calls = []
    with patch("shared.http_retry.requests.get", side_effect=requests.exceptions.ReadTimeout("x")):
        with patch.object(http_retry_module.time, "sleep", side_effect=sleep_calls.append):
            with pytest.raises(requests.exceptions.ReadTimeout):
                get_with_retry("https://example.com", max_retries=3, backoff_seconds=1.0)

    assert sleep_calls == [1.0, 2.0]