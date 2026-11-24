"""CLI: python -m krxreport {daily,backfill,probe}"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from . import pipeline
from .indicators import DataQualityError
from .krx_client import KrxApiError, KrxClient
from .parsing import SchemaError

log = logging.getLogger("krxreport")
KST = ZoneInfo("Asia/Seoul")
EXIT_DATA_PROBLEM = 2


def _today_kst():
    return datetime.now(KST).date()


def _parse_date(text: str):
    return datetime.strptime(text, "%Y%m%d").date()


def _github_output(**values: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def cmd_daily(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        result = pipeline.run_daily(KrxClient(), root, _today_kst(), args.date)
    except (DataQualityError, SchemaError) as exc:
        problem = root / "quality_error.md"
        problem.write_text(f"## 데이터 품질 검사 실패\n\n```\n{exc}\n```\n\n리포트를 발행하지 않았습니다.\n",
                           encoding="utf-8")
        log.error("data problem: %s", exc)
        _github_output(status="quality_error", date=args.date or "")
        return EXIT_DATA_PROBLEM
    log.info("status=%s date=%s", result.status, result.date)
    _github_output(status=result.status, date=result.date or "",
                   report=str(result.markdown_path or ""))
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    store = pipeline.Store(Path(args.root) / "data")
    kinds = tuple(args.kinds.split(","))
    saved = pipeline.backfill(KrxClient(), store, _parse_date(args.start), _parse_date(args.end), kinds)
    log.info("saved %d trading days", saved)
    return 0


def cmd_reports(args: argparse.Namespace) -> int:
    """Rebuild past reports from saved data (no API calls, no Issues)."""
    result = pipeline.build_range(Path(args.root), args.start, args.end)
    log.info("built %d reports, skipped %d", len(result.built), len(result.failed))
    for day, reason in result.failed.items():
        log.info("  %s: %s", day, reason)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Show what KRX actually returns so you can fix field names in config.py."""
    client = KrxClient()
    for kind in pipeline.KINDS:
        rows = client.fetch(kind, args.date)
        log.info("=== %s: %d rows", kind, len(rows))
        if not rows:
            continue
        log.info("fields: %s", sorted(rows[0]))
        log.info("first row: %s", json.dumps(rows[0], ensure_ascii=False))
        if "PROD_NM" in rows[0]:
            products = sorted({r["PROD_NM"] for r in rows})
            log.info("PROD_NM values: %s", products)
            sample = [r.get("ISU_NM") for r in rows[:5]]
            log.info("ISU_NM samples: %s", sample)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()  # local runs read KRX_API_KEY from .env; on GitHub it comes from Secrets
    parser = argparse.ArgumentParser(prog="krxreport")
    parser.add_argument("--root", default=".", help="project root (data/, reports/, docs/)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    daily = sub.add_parser("daily", help="build the report for the latest trading day")
    daily.add_argument("--date", help="YYYYMMDD; default: latest trading day")
    daily.set_defaults(fn=cmd_daily)

    fill = sub.add_parser("backfill", help="download past days for the backtest")
    fill.add_argument("--start", required=True)
    fill.add_argument("--end", required=True)
    fill.add_argument("--kinds", default="etf", help="comma list of etf,futures,options")
    fill.set_defaults(fn=cmd_backfill)

    past = sub.add_parser("reports", help="rebuild past reports from saved data")
    past.add_argument("--start", required=True)
    past.add_argument("--end", required=True)
    past.set_defaults(fn=cmd_reports)

    probe = sub.add_parser("probe", help="print real KRX field names for one day")
    probe.add_argument("--date", required=True)
    probe.set_defaults(fn=cmd_probe)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except KrxApiError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
