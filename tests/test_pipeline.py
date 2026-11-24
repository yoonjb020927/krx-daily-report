from datetime import datetime, timedelta

import pandas as pd

from krxreport import pipeline
from krxreport.storage import Store


def _day_after(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d") + timedelta(days=1)


def _backfill_all(source, root):
    store = Store(root / "data")
    first, last = source.days[0], source.days[-2]
    pipeline.backfill(source, store, datetime.strptime(first, "%Y%m%d").date(),
                      datetime.strptime(last, "%Y%m%d").date(), ("etf",))
    return store


def test_daily_run_writes_markdown_and_pages(source, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    _backfill_all(source, tmp_path)
    today = _day_after(source.days[-1]).date()

    result = pipeline.run_daily(source, tmp_path, today)

    assert result.status == "ok" and result.date == source.days[-1]
    md = result.markdown_path.read_text(encoding="utf-8")
    for section in ("시장 요약", "PCR (거래량)", "Max Pain", "시그널 백테스트 검증", "투자 권유가 아닙니다"):
        assert section in md
    assert (tmp_path / "docs" / "index.html").exists()
    assert (tmp_path / "docs" / "reports" / f"{result.date}.html").exists()
    summary = pd.read_csv(tmp_path / "summary.csv", dtype={"date": str})
    assert summary["date"].tolist() == [result.date]


def test_second_run_same_day_is_skipped(source, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    _backfill_all(source, tmp_path)
    today = _day_after(source.days[-1]).date()
    pipeline.run_daily(source, tmp_path, today)
    assert pipeline.run_daily(source, tmp_path, today).status == "skipped"


def test_latest_trading_date_skips_weekend_and_holiday(source):
    last = source.days[-1]
    source.days.remove(last)
    source.holidays.add(last)           # KRX still returns rows, but with blank prices
    today = _day_after(last).date()
    assert source.fetch("etf", last)    # not empty...
    assert pipeline.find_latest_trading_date(source, today) == source.days[-1]


def test_build_range_makes_one_report_per_day_without_peeking_ahead(source, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    store = _backfill_all(source, tmp_path)
    days = source.days[-6:-1]
    for d in days:
        pipeline.ingest_day(source, store, d, ("futures", "options"))
    # newest report already exists (made by the daily job)
    pipeline.run_daily(source, tmp_path, _day_after(source.days[-1]).date(), source.days[-1])
    front = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")

    result = pipeline.build_range(tmp_path, days[0], days[-1])

    assert result.built == days and not result.failed
    for d in days:
        assert (tmp_path / "reports" / f"{d}.md").exists()
    # a past report's trend table must not show later days
    first = (tmp_path / "reports" / f"{days[0]}.md").read_text(encoding="utf-8")
    later = f"{days[1][:4]}-{days[1][4:6]}-{days[1][6:]}"
    assert later not in first
    # front page still shows the newest report
    assert (tmp_path / "docs" / "index.html").read_text(encoding="utf-8") == front


def test_holiday_is_not_saved(source, tmp_path):
    holiday = source.days[3]
    source.holidays.add(holiday)
    store = Store(tmp_path)
    assert pipeline.ingest_day(source, store, holiday) is False
    assert not any(store.has(kind, holiday) for kind in pipeline.KINDS)
    assert store.is_holiday(holiday)


def test_backfill_rerun_makes_no_api_calls_for_saved_days_and_holidays(source, tmp_path, monkeypatch):
    from datetime import date as Date
    monkeypatch.setattr(pipeline.config, "BACKFILL_SLEEP_SEC", 0)
    source.holidays.add(source.days[2])
    store = Store(tmp_path)
    start = datetime.strptime(source.days[0], "%Y%m%d").date()
    end = datetime.strptime(source.days[9], "%Y%m%d").date()
    pipeline.backfill(source, store, start, end, ("etf",))
    source.calls.clear()

    pipeline.backfill(source, store, start, end, ("etf",))   # e.g. daily run with a warm cache

    assert source.calls == []
    assert isinstance(start, Date)


def test_panel_drops_holiday_snapshots_already_on_disk(source, tmp_path):
    from krxreport import parsing, signals
    days = source.days[:3]
    source.holidays.add(days[1])
    history = pd.concat([parsing.normalize("etf", source.fetch("etf", d)) for d in days])
    panel = signals.build_panel(history)
    assert panel["close"].index.tolist() == [days[0], days[2]]


def test_store_files_are_per_day_and_reloadable(source, tmp_path):
    store = Store(tmp_path)
    day = source.days[0]
    pipeline.ingest_day(source, store, day)
    assert store.path("etf", day).name == f"{day}.csv.gz"
    loaded = store.load_day("etf", day)
    assert loaded["code"].iloc[0] == "000000"   # leading zeros kept
