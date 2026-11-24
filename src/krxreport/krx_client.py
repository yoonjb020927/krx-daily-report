"""Thin client for the KRX Open API."""
from __future__ import annotations

import logging
import os
import time
from typing import Protocol

import requests

from . import config

log = logging.getLogger(__name__)


class KrxApiError(RuntimeError):
    """Raised when KRX returns an error or an unexpected payload."""


class RowSource(Protocol):
    def fetch(self, kind: str, date: str) -> list[dict]: ...


class KrxClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None) -> None:
        key = api_key or os.environ.get("KRX_API_KEY")
        if not key:
            raise KrxApiError("KRX_API_KEY is not set. Put it in .env or GitHub Secrets.")
        self._key = key
        self._session = session or requests.Session()

    def fetch(self, kind: str, date: str) -> list[dict]:
        """Return raw rows for one trading day. Empty list means no trading (holiday)."""
        if kind not in config.ENDPOINTS:
            raise ValueError(f"unknown kind: {kind}")
        url = f"{config.BASE_URL}/{config.ENDPOINTS[kind]}"
        payload = self._get_with_retry(url, {"basDd": date})
        rows = payload.get(config.RESPONSE_ROWS_KEY)
        if rows is None:
            raise KrxApiError(f"{kind} {date}: no '{config.RESPONSE_ROWS_KEY}' in response: {str(payload)[:300]}")
        return rows

    def _get_with_retry(self, url: str, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, config.REQUEST_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    headers={config.AUTH_HEADER: self._key},
                    params=params,
                    timeout=config.REQUEST_TIMEOUT_SEC,
                )
                if resp.status_code in (401, 403):
                    raise KrxApiError(
                        f"HTTP {resp.status_code}: key rejected or service not approved for {url}"
                    )
                resp.raise_for_status()
                return resp.json()
            except KrxApiError:
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                log.warning("KRX request failed (attempt %d): %s", attempt, exc)
                time.sleep(2 * attempt)
        raise KrxApiError(f"KRX request failed after retries: {last_error}")
