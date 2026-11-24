from datetime import date

import pytest
import requests

from krxreport import __main__ as cli
from krxreport import config, pipeline
from krxreport.krx_client import KrxApiError, KrxClient


class FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def get(self, url, headers, params, timeout):
        self.requests.append({"url": url, "headers": headers, "params": params})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("krxreport.krx_client.time.sleep", lambda _: None)


def test_client_sends_key_and_date():
    session = FakeSession([FakeResponse(200, {"OutBlock_1": [{"A": 1}]})])
    rows = KrxClient("k", session).fetch("etf", "20261123")
    assert rows == [{"A": 1}]
    sent = session.requests[0]
    assert sent["url"].endswith("etp/etf_bydd_trd")
    assert sent["headers"] == {config.AUTH_HEADER: "k"}
    assert sent["params"] == {"basDd": "20261123"}


def test_client_retries_transient_errors():
    session = FakeSession([requests.ConnectionError("boom"), FakeResponse(200, {"OutBlock_1": []})])
    assert KrxClient("k", session).fetch("options", "20261123") == []


def test_client_gives_up_after_retries():
    session = FakeSession([requests.Timeout("slow")] * config.REQUEST_RETRIES)
    with pytest.raises(KrxApiError, match="after retries"):
        KrxClient("k", session).fetch("etf", "20261123")


def test_client_reports_unapproved_service():
    session = FakeSession([FakeResponse(401, {})])
    with pytest.raises(KrxApiError, match="not approved"):
        KrxClient("k", session).fetch("futures", "20261123")


def test_client_rejects_unexpected_payload():
    session = FakeSession([FakeResponse(200, {"respMsg": "error"})])
    with pytest.raises(KrxApiError, match="OutBlock_1"):
        KrxClient("k", session).fetch("etf", "20261123")


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("KRX_API_KEY", raising=False)
    with pytest.raises(KrxApiError, match="KRX_API_KEY"):
        KrxClient()


def test_cli_daily_writes_quality_error_file(source, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "KrxClient", lambda: source)
    monkeypatch.setattr(cli, "_today_kst", lambda: date(2030, 1, 1))
    monkeypatch.setattr(config, "OPTIONS_PRODUCT", "없는상품")
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))

    code = cli.main(["--root", str(tmp_path), "daily", "--date", source.days[-1]])

    assert code == cli.EXIT_DATA_PROBLEM
    assert "데이터 품질 검사 실패" in (tmp_path / "quality_error.md").read_text(encoding="utf-8")
    assert "status=quality_error" in out.read_text(encoding="utf-8")


def test_cli_daily_ok_sets_outputs(source, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "KrxClient", lambda: source)
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert cli.main(["--root", str(tmp_path), "daily", "--date", source.days[-1]]) == 0
    assert f"date={source.days[-1]}" in out.read_text(encoding="utf-8")


def test_cli_probe_and_backfill(source, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cli, "KrxClient", lambda: source)
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    caplog.set_level("INFO")
    assert cli.main(["probe", "--date", source.days[0]]) == 0
    assert "PROD_NM values" in caplog.text
    assert cli.main(["--root", str(tmp_path), "backfill", "--start", source.days[0],
                     "--end", source.days[4]]) == 0
    assert len(list((tmp_path / "data" / "etf").glob("*/*.csv.gz"))) == 5
