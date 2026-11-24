"""Fetch -> store -> compute -> render. The CLI and the tests both drive this."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import config, indicators, parsing, report, signals
from .krx_client import RowSource
from .storage import Store

log = logging.getLogger(__name__)
KINDS = ("etf", "futures", "options")
SUMMARY_FILE = "summary.csv"
CLOSE_FIELD = "TDD_CLSPRC"


@dataclass(frozen=True)
class DailyResult:
    status: str          # "ok" | "skipped"
    date: str | None
    markdown_path: Path | None = None


def _yyyymmdd(d: Date) -> str:
    return d.strftime("%Y%m%d")


def is_trading_day(rows: list[dict]) -> bool:
    """KRX answers holidays with the full instrument list but blank prices."""
    return any(str(r.get(CLOSE_FIELD, "")).strip() not in ("", "-") for r in rows)


def find_latest_trading_date(source: RowSource, today: Date) -> str | None:
    """Most recent past day with real ETF prices (KRX publishes T+1)."""
    for back in range(1, config.LOOKBACK_DAYS_FOR_LATEST + 1):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        if is_trading_day(source.fetch("etf", _yyyymmdd(day))):
            return _yyyymmdd(day)
    return None


def ingest_day(source: RowSource, store: Store, day: str, kinds: tuple[str, ...] = KINDS) -> bool:
    """Download and save one day. Returns False (and saves nothing) on a holiday."""
    if store.is_holiday(day):
        return False
    fetched = {}
    for kind in kinds:
        if store.has(kind, day):
            continue
        rows = source.fetch(kind, day)
        if not is_trading_day(rows):
            if rows:                       # listed but no prices = market closed
                store.mark_holiday(day)
            return False
        fetched[kind] = rows
    for kind, rows in fetched.items():
        store.save(kind, day, parsing.normalize(kind, rows))
    return True


def backfill(source: RowSource, store: Store, start: Date, end: Date, kinds: tuple[str, ...]) -> int:
    saved = 0
    day = start
    while day <= end:
        key = _yyyymmdd(day)
        already = store.is_holiday(key) or all(store.has(k, key) for k in kinds)
        if day.weekday() < 5 and not already:
            if ingest_day(source, store, key, kinds):
                saved += 1
                log.info("saved %s", key)
            time.sleep(config.BACKFILL_SLEEP_SEC)   # only after real API calls
        day += timedelta(days=1)
    return saved


def _check_etf(etf_today: pd.DataFrame) -> None:
    if len(etf_today) < config.LIMITS.min_etf_rows:
        raise indicators.DataQualityError(f"only {len(etf_today)} ETF rows (expected ≥ {config.LIMITS.min_etf_rows})")
    if etf_today["close"].isna().mean() > 0.2:
        raise indicators.DataQualityError("over 20% of ETF closes are missing; check ETF_FIELDS")


def _update_summary(root: Path, day: str, fut: indicators.FuturesSummary, opt: indicators.OptionsSummary) -> pd.DataFrame:
    path = root / SUMMARY_FILE
    row = pd.DataFrame([{
        "date": day, "futures_close": fut.close, "basis": fut.basis,
        "pcr_volume": opt.pcr_volume, "pcr_open_interest": opt.pcr_open_interest, "max_pain": opt.max_pain,
    }])
    old = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame()
    merged = pd.concat([old, row], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    merged.to_csv(path, index=False)
    return merged


def build_report(store: Store, root: Path, day: str,
                 etf_history: pd.DataFrame | None = None) -> report.ReportData:
    """Report as it would have looked on `day`: only data up to that day is used."""
    etf_today = store.load_day("etf", day)
    _check_etf(etf_today)
    fut = indicators.summarize_futures(parsing.derive("futures", store.load_day("futures", day)))
    opt = indicators.summarize_options(parsing.derive("options", store.load_day("options", day)))

    history = etf_history if etf_history is not None else store.load_history("etf")
    panel = signals.build_panel(history[history["date"] <= day])
    stats = signals.evaluate(panel)
    candidates = signals.todays_candidates(panel, etf_today, stats)
    summary = _update_summary(root, day, fut, opt)
    return report.ReportData(day, fut, opt, stats, candidates, summary[summary["date"] <= day])


@dataclass(frozen=True)
class RangeResult:
    built: list[str]
    failed: dict[str, str]     # date -> reason


def build_range(root: Path, start: str, end: str) -> RangeResult:
    """Rebuild reports for past days that have ETF, futures and options data saved."""
    store = Store(root / "data")
    days = [d for d in store.dates("options")
            if start <= d <= end and store.has("etf", d) and store.has("futures", d)]
    history = store.load_history("etf")
    built, failed = [], {}
    for day in days:
        try:
            write_outputs(root, build_report(store, root, day, history))
            built.append(day)
            log.info("built %s", day)
        except (indicators.DataQualityError, parsing.SchemaError) as exc:
            failed[day] = str(exc)
            log.warning("skipped %s: %s", day, exc)
    return RangeResult(built, failed)


def write_outputs(root: Path, data: report.ReportData) -> Path:
    md_path = root / "reports" / f"{data.date}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report.render_markdown(data), encoding="utf-8")

    docs = root / "docs"
    (docs / "reports").mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").touch()
    archive = sorted(p.stem for p in (root / "reports").glob("*.md"))
    (docs / "reports" / f"{data.date}.html").write_text(
        report.render_html(data, archive, link_prefix=""), encoding="utf-8")
    if data.date == archive[-1]:          # rebuilding an older day must not replace the front page
        (docs / "index.html").write_text(report.render_html(data, archive), encoding="utf-8")
    return md_path


def run_daily(source: RowSource, root: Path, today: Date, day: str | None = None) -> DailyResult:
    store = Store(root / "data")
    target = day or find_latest_trading_date(source, today)
    if target is None:
        return DailyResult("skipped", None)
    if (root / "reports" / f"{target}.md").exists() and day is None:
        log.info("report for %s already exists", target)
        return DailyResult("skipped", target)
    if not ingest_day(source, store, target):
        return DailyResult("skipped", target)
    data = build_report(store, root, target)
    return DailyResult("ok", target, write_outputs(root, data))
