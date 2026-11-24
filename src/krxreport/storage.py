"""One gzip CSV per kind per trading day: data/<kind>/<YYYY>/<YYYYMMDD>.csv.gz.

Files are written once and never rewritten, so the git history stays small.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_DTYPES = {"date": str, "code": str, "name": str, "product": str, "right": str, "expiry": str,
               "session": str}


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, kind: str, date: str) -> Path:
        return self.root / kind / date[:4] / f"{date}.csv.gz"

    def has(self, kind: str, date: str) -> bool:
        return self.path(kind, date).exists()

    def save(self, kind: str, date: str, frame: pd.DataFrame) -> Path:
        target = self.path(kind, date)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, compression="gzip")
        return target

    def load_day(self, kind: str, date: str) -> pd.DataFrame:
        return pd.read_csv(self.path(kind, date), dtype=TEXT_DTYPES, compression="gzip")

    @property
    def _holiday_file(self) -> Path:
        return self.root / "holidays.txt"

    def is_holiday(self, date: str) -> bool:
        return self._holiday_file.exists() and date in self._holiday_file.read_text().split()

    def mark_holiday(self, date: str) -> None:
        if self.is_holiday(date):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self._holiday_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{date}\n")

    def dates(self, kind: str) -> list[str]:
        return sorted(p.name[:8] for p in (self.root / kind).glob("*/*.csv.gz"))

    def load_history(self, kind: str, last_n_days: int | None = None) -> pd.DataFrame:
        days = self.dates(kind)
        if last_n_days is not None:
            days = days[-last_n_days:]
        frames = [self.load_day(kind, d) for d in days]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
