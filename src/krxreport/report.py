"""Render the daily report as Markdown (repo + Issue) and HTML (GitHub Pages)."""
from __future__ import annotations

import html
import math
from dataclasses import dataclass

import pandas as pd

from . import config
from .indicators import FuturesSummary, OptionsSummary, max_pain_gap
from .signals import SignalStats

DISCLAIMER = "본 리포트는 과거 데이터 기반 자동 생성 자료이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다."


@dataclass(frozen=True)
class ReportData:
    date: str
    futures: FuturesSummary
    options: OptionsSummary
    stats: list[SignalStats]
    candidates: pd.DataFrame
    summary_history: pd.DataFrame   # date, pcr_volume, pcr_open_interest, max_pain, futures_close


def _fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _max_pain_note(o: OptionsSummary, f: FuturesSummary) -> str:
    note = f"{o.front_expiry} 만기 기준, 선물 대비 {o.max_pain - f.close:+,.1f}"
    gap = max_pain_gap(o, f)
    if gap > config.LIMITS.max_pain_warn_gap:
        note += f" ⚠ 괴리 {gap:.0%}: 급등락으로 포지션이 가격을 못 따라간 상태일 수 있음"
    return note


def _pct_or_dash(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:+.2%}"


METHOD_NOTE = (
    "초과수익 = 시그널 다음 날 시가에 사서 5영업일 뒤 종가에 판 수익 − 같은 날 유동성 상위 ETF 전체를 "
    "같은 방식으로 샀을 때의 평균 수익. 채택 조건: t값 ≥ 2, 최근 30% 구간에서도 초과수익 > 0, 비용 차감 후 수익 > 0."
)


def _sections(data: ReportData) -> list[tuple[str, list[str], list[list[str]]]]:
    """(title, header, rows) tables shared by both renderers."""
    f, o = data.futures, data.options
    market = [
        ["코스피200 선물(최근월)", f"{f.close:,.2f}", f.name],
        ["현물지수", f"{f.spot:,.2f}", ""],
        ["베이시스", f"{f.basis:+,.2f}", "선물 - 현물"],
        ["선물 미결제약정", f"{f.open_interest:,.0f}", ""],
        ["PCR (거래량)", f"{o.pcr_volume:.2f}", f"풋 {o.put_volume:,.0f} / 콜 {o.call_volume:,.0f}"],
        ["PCR (미결제약정)", f"{o.pcr_open_interest:.2f}", ""],
        ["Max Pain", f"{o.max_pain:,.1f}", _max_pain_note(o, f)],
    ]
    stats = [
        [s.label, f"{s.signal_days:,} / {s.trades:,}", f"{s.avg_excess:+.2%}", f"{s.t_stat:.2f}",
         _pct_or_dash(s.oos_excess), f"{s.avg_return:+.2%}", "✅ 사용" if s.is_valid else "❌ 제외"]
        for s in data.stats
    ]
    picks = [
        [r.name, r.code, f"{r.close:,.0f}", r.signal, f"{r.avg_excess:+.2%}", f"{r.t_stat:.2f}"]
        for r in data.candidates.itertuples()
    ]
    hist = data.summary_history.tail(10).iloc[::-1]
    trend = [
        [_fmt_date(str(r.date)), f"{r.pcr_volume:.2f}", f"{r.pcr_open_interest:.2f}", f"{r.max_pain:,.1f}",
         f"{r.futures_close:,.2f}"]
        for r in hist.itertuples()
    ]
    return [
        ("시장 요약", ["항목", "값", "비고"], market),
        ("시그널 백테스트 검증", ["시그널", "발생일 / 거래 수", "초과수익", "t값", "최근 30% 초과수익",
                              "평균 수익(비용 차감)", "채택"], stats),
        ("매매 후보 (검증 통과 시그널)", ["ETF", "코드", "종가", "시그널", "과거 초과수익", "t값"], picks),
        ("최근 10영업일 추이", ["날짜", "PCR(거래량)", "PCR(미결제)", "Max Pain", "선물 종가"], trend),
    ]


def render_markdown(data: ReportData) -> str:
    lines = [f"# KRX 데일리 리포트 · {_fmt_date(data.date)}", ""]
    for title, header, rows in _sections(data):
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["_해당 없음_", ""]
            continue
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        lines.append("")
    lines += [f"_{METHOD_NOTE}_", "", f"> {DISCLAIMER}", ""]
    return "\n".join(lines)


def render_html(data: ReportData, archive: list[str], link_prefix: str = "reports/") -> str:
    esc = html.escape
    body = []
    for title, header, rows in _sections(data):
        body.append(f"<h2>{esc(title)}</h2>")
        if not rows:
            body.append("<p class='muted'>해당 없음</p>")
            continue
        head = "".join(f"<th>{esc(h)}</th>" for h in header)
        trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
        body.append(f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table></div>")
    links = "".join(
        f"<li><a href='{link_prefix}{d}.html'>{_fmt_date(d)}</a></li>" for d in sorted(archive, reverse=True)[:60]
    )
    return _PAGE.format(
        title=f"KRX 데일리 리포트 · {_fmt_date(data.date)}",
        body="\n".join(body),
        archive=links,
        disclaimer=esc(METHOD_NOTE) + "</p><p>" + esc(DISCLAIMER),
    )


_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRX 데일리 리포트</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --head:#f5f5f5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eaeaea; --muted:#999; --line:#333; --head:#1f1f1f; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:system-ui,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
       max-width:960px; margin:0 auto; padding:24px 16px; line-height:1.5; }}
h1 {{ font-size:1.5rem; }} h2 {{ font-size:1.15rem; margin-top:2rem; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; white-space:nowrap; }}
th {{ background:var(--head); }}
.muted, footer {{ color:var(--muted); font-size:.9rem; }}
</style></head><body>
<h1>{title}</h1>
{body}
<h2>지난 리포트</h2><ul>{archive}</ul>
<footer><p>{disclaimer}</p></footer>
</body></html>
"""
