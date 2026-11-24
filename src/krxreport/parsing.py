"""Turn raw KRX rows into typed DataFrames."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

TEXT_COLUMNS = {"date", "code", "name", "product", "right"}
REGULAR, NIGHT = "regular", "night"
NIGHT_MARK = "(야간)"
# strikes ≥ 1000 come with a thousands comma: "코스피200 C 202612 1,002.5 (정규)"
_OPTION_NAME = re.compile(r"\b([CP])\s+(\d{6})\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?![\d,])")
_FUTURE_NAME = re.compile(r"\bF\s+(\d{6})")
_RIGHT_ALIASES = {"CALL": "C", "콜": "C", "C": "C", "PUT": "P", "풋": "P", "P": "P"}


class SchemaError(ValueError):
    """Raised when KRX rows lack the fields we expect (see config.py)."""


def to_number(value: object) -> float:
    """KRX sends numbers as strings like '1,234' or '-'. Convert safely."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if text in ("", "-"):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize(kind: str, rows: list[dict]) -> pd.DataFrame:
    fields = config.FIELDS_BY_KIND[kind]
    columns = list(fields.values())
    if not rows:
        return pd.DataFrame(columns=columns)
    missing = [raw for raw in fields if raw not in rows[0]]
    if missing:
        raise SchemaError(
            f"{kind}: fields {missing} not in KRX response. "
            f"Available: {sorted(rows[0])}. Fix config.{kind.upper()}_FIELDS."
        )
    frame = pd.DataFrame([{ours: row[raw] for raw, ours in fields.items()} for row in rows])
    numeric = {c: frame[c].map(to_number) for c in columns if c not in TEXT_COLUMNS}
    return derive(kind, frame.assign(**numeric))


def derive(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Columns computed from the contract name. Re-run on load so parser fixes
    also apply to files saved earlier."""
    if kind == "options":
        frame = _add_option_terms(frame)
    if kind == "futures":
        frame = frame.assign(expiry=frame["name"].map(_future_expiry))
    if kind in ("options", "futures"):
        frame = frame.assign(session=frame["name"].map(_session))
    return frame


def _session(name: str) -> str:
    """KRX lists each derivative twice: '... (정규)'/'(주간)' and '... (야간)'."""
    return NIGHT if NIGHT_MARK in str(name) else REGULAR


def _future_expiry(name: str) -> str | None:
    match = _FUTURE_NAME.search(str(name))
    return match.group(1) if match else None


def _add_option_terms(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["name"].map(lambda n: _OPTION_NAME.search(str(n)))
    from_name_right = parsed.map(lambda m: m.group(1) if m else None)
    right = frame["right"].map(lambda r: _RIGHT_ALIASES.get(str(r).strip().upper()))
    return frame.assign(
        right=right.fillna(from_name_right),
        expiry=parsed.map(lambda m: m.group(2) if m else None),
        strike=parsed.map(lambda m: to_number(m.group(3)) if m else np.nan),
    )
