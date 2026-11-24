"""Synthetic KRX-shaped responses so tests run without an API key."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest


BLANK_ON_HOLIDAY = {"TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC", "NAV"}


def fmt(x: float, decimals: int = 0) -> str:
    """KRX style: comma-separated string."""
    return f"{x:,.{decimals}f}"


class FakeSource:
    """Deterministic random-walk market. Weekends return no rows, like KRX."""

    def __init__(self, start: date, n_days: int, n_etfs: int = 120, holidays: set[str] | None = None) -> None:
        rng = np.random.default_rng(7)
        self.holidays = holidays or set()
        self.days = []
        d = start
        while len(self.days) < n_days:
            key = d.strftime("%Y%m%d")
            if d.weekday() < 5 and key not in self.holidays:
                self.days.append(key)
            d += timedelta(days=1)
        steps = rng.normal(0.0005, 0.015, size=(n_days, n_etfs))
        self.close = 10_000 * np.exp(np.cumsum(steps, axis=0))
        self.open = self.close * (1 + rng.normal(0, 0.003, size=steps.shape))
        self.value = rng.lognormal(20, 1, size=steps.shape)
        self.n_etfs = n_etfs
        self.calls: list[tuple[str, str]] = []

    def fetch(self, kind: str, day: str) -> list[dict]:
        self.calls.append((kind, day))
        builders = {"etf": self._etf, "futures": self._futures, "options": self._options}
        if day in self.holidays:
            # real KRX: holidays return the instrument list with blank prices
            return [{k: ("" if k in BLANK_ON_HOLIDAY else v) for k, v in row.items()}
                    for row in builders[kind](day, 0)]
        if day not in self.days:
            return []
        return builders[kind](day, self.days.index(day))

    def _etf(self, day: str, i: int) -> list[dict]:
        return [
            {
                "BAS_DD": day, "ISU_CD": f"{j:06d}", "ISU_NM": f"테스트ETF{j}",
                "TDD_OPNPRC": fmt(self.open[i, j]), "TDD_HGPRC": fmt(self.close[i, j] * 1.01),
                "TDD_LWPRC": fmt(self.close[i, j] * 0.99), "TDD_CLSPRC": fmt(self.close[i, j]),
                "ACC_TRDVOL": fmt(self.value[i, j] / self.close[i, j]), "ACC_TRDVAL": fmt(self.value[i, j]),
                "NAV": fmt(self.close[i, j], 2),
            }
            for j in range(self.n_etfs)
        ]

    def _futures(self, day: str, i: int) -> list[dict]:
        rows = []
        for prod, label, expiry, px in (
            ("코스피200 선물", "코스피200 F 202612 (주간)", "202612", 351.2),
            ("코스피200 선물", "코스피200 F 202612 (야간)", "202612", 999.0),   # night session: ignore
            ("코스피200 선물", "코스피200 F 202703 (주간)", "202703", 352.9),
            ("코스피200 선물", "코스피200 SP 2612-2703 (주간)", "202612", -1.7),  # spread: ignore
            ("미니코스피200 선물", "미니코스피200 F 202612 (주간)", "202612", 351.1),
        ):
            rows.append({
                "BAS_DD": day, "PROD_NM": prod, "ISU_CD": f"F{expiry}", "ISU_NM": label,
                "TDD_CLSPRC": fmt(px, 2), "SPOT_PRC": "350.00", "ACC_TRDVOL": "200,000",
                "ACC_OPNINT_QTY": "310,000",
            })
        return rows

    def _options(self, day: str, i: int) -> list[dict]:
        rows = []
        for expiry in ("202612", "202701"):
            for strike in np.arange(320, 382.5, 2.5):
                for right, call_put in (("CALL", "C"), ("PUT", "P")):
                    # put OI concentrated below 345, call OI above 355 -> max pain near 350
                    heavy = (call_put == "P" and strike <= 345) or (call_put == "C" and strike >= 355)
                    base = {"BAS_DD": day, "PROD_NM": "코스피200 옵션", "RGHT_TP_NM": right,
                            "ISU_CD": f"{call_put}{expiry}{strike}", "TDD_CLSPRC": "1.25"}
                    # real KRX format: multiple spaces before strike, session suffix
                    rows.append({**base, "ISU_NM": f"코스피200 {call_put} {expiry}   {strike:.1f} (정규)",
                                 "ACC_TRDVOL": fmt(10_000 if heavy else 3_000),
                                 "ACC_OPNINT_QTY": fmt(20_000 if heavy else 2_000)})
                    # night duplicate with skewed numbers: must be ignored
                    rows.append({**base, "ISU_NM": f"코스피200 {call_put} {expiry}   {strike:.1f} (야간)",
                                 "ACC_TRDVOL": fmt(50_000 if call_put == "P" else 1),
                                 "ACC_OPNINT_QTY": fmt(90_000 if call_put == "P" else 1)})
        # distractor product: tiny call volume, huge put volume
        rows += [
            {"BAS_DD": day, "PROD_NM": "미니코스피200 옵션", "RGHT_TP_NM": "CALL", "ISU_CD": "MC",
             "ISU_NM": "미니코스피200 C 202612 350.0", "TDD_CLSPRC": "1", "ACC_TRDVOL": "10",
             "ACC_OPNINT_QTY": "10"},
            {"BAS_DD": day, "PROD_NM": "미니코스피200 옵션", "RGHT_TP_NM": "PUT", "ISU_CD": "MP",
             "ISU_NM": "미니코스피200 P 202612 350.0", "TDD_CLSPRC": "1", "ACC_TRDVOL": "9,999,999",
             "ACC_OPNINT_QTY": "9,999,999"},
        ]
        return rows


@pytest.fixture
def source() -> FakeSource:
    return FakeSource(start=date(2025, 1, 2), n_days=260)
