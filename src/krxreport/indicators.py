"""Derivatives indicators: PCR, Max Pain, front-month futures."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .parsing import REGULAR


class DataQualityError(RuntimeError):
    """Raised when a computed value is outside a plausible range."""


@dataclass(frozen=True)
class OptionsSummary:
    pcr_volume: float
    pcr_open_interest: float
    front_expiry: str
    max_pain: float
    call_volume: float
    put_volume: float


@dataclass(frozen=True)
class FuturesSummary:
    name: str
    close: float
    spot: float
    basis: float
    open_interest: float


def kospi200_options(options: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session KOSPI200 option series with a parsed strike."""
    mask = (
        (options["product"].str.strip() == config.OPTIONS_PRODUCT)
        & (options["session"] == REGULAR)
        & options["right"].isin(["C", "P"])
        & options["strike"].notna()
    )
    return options.loc[mask]


def put_call_ratio(options: pd.DataFrame, column: str) -> float:
    calls = options.loc[options["right"] == "C", column].sum()
    puts = options.loc[options["right"] == "P", column].sum()
    if calls <= 0:
        raise DataQualityError(f"call {column} is {calls}; cannot compute PCR")
    return float(puts / calls)


def max_pain(options: pd.DataFrame) -> float:
    """Strike where total intrinsic value owed to option holders is smallest."""
    by_strike = options.pivot_table(
        index="strike", columns="right", values="open_interest", aggfunc="sum", fill_value=0
    ).reindex(columns=["C", "P"], fill_value=0)
    strikes = by_strike.index.to_numpy(dtype=float)
    call_oi = by_strike["C"].to_numpy(dtype=float)
    put_oi = by_strike["P"].to_numpy(dtype=float)
    settle = strikes[:, None]
    pain = (np.clip(settle - strikes, 0, None) * call_oi).sum(axis=1) + (
        np.clip(strikes - settle, 0, None) * put_oi
    ).sum(axis=1)
    return float(strikes[int(np.argmin(pain))])


def summarize_options(options: pd.DataFrame) -> OptionsSummary:
    k200 = kospi200_options(options)
    if len(k200) < config.LIMITS.min_option_rows:
        raise DataQualityError(
            f"only {len(k200)} '{config.OPTIONS_PRODUCT}' rows; check OPTIONS_PRODUCT / name parsing"
        )
    dupes = k200.duplicated(["expiry", "strike", "right"]).sum()
    if dupes:
        raise DataQualityError(f"{dupes} duplicate (expiry, strike, right) rows; option name parsing is off")
    # On expiry day the expiring series still lists but with zero open interest:
    # use the nearest expiry that still has positions.
    oi_by_expiry = k200.groupby("expiry")["open_interest"].sum()
    live = oi_by_expiry[oi_by_expiry > 0]
    if live.empty:
        raise DataQualityError("no option series with open interest")
    front = str(live.index.min())
    summary = OptionsSummary(
        pcr_volume=put_call_ratio(k200, "volume"),
        pcr_open_interest=put_call_ratio(k200, "open_interest"),
        front_expiry=front,
        max_pain=max_pain(k200.loc[k200["expiry"] == front]),
        call_volume=float(k200.loc[k200["right"] == "C", "volume"].sum()),
        put_volume=float(k200.loc[k200["right"] == "P", "volume"].sum()),
    )
    _check_pcr(summary)
    return summary


def _check_pcr(summary: OptionsSummary) -> None:
    lo, hi = config.LIMITS.pcr_min, config.LIMITS.pcr_max
    for label, value in (("volume", summary.pcr_volume), ("OI", summary.pcr_open_interest)):
        if not lo <= value <= hi:
            raise DataQualityError(
                f"PCR({label})={value:.2f} outside [{lo}, {hi}] "
                f"(calls={summary.call_volume:,.0f}, puts={summary.put_volume:,.0f})"
            )


def max_pain_gap(options: OptionsSummary, futures: FuturesSummary) -> float:
    """|Max Pain - futures| / futures. Large gaps happen in sharp moves (positions
    lag the price), so this is shown as a warning, not treated as bad data."""
    return abs(options.max_pain - futures.close) / futures.close


def summarize_futures(futures: pd.DataFrame) -> FuturesSummary:
    k200 = futures.loc[
        (futures["product"].str.strip() == config.FUTURES_PRODUCT)
        & (futures["session"] == REGULAR)
        & futures["expiry"].notna()
    ]
    if k200.empty:
        raise DataQualityError(f"no '{config.FUTURES_PRODUCT}' rows; check FUTURES_PRODUCT")
    front = k200.sort_values("expiry").iloc[0]
    return FuturesSummary(
        name=str(front["name"]),
        close=float(front["close"]),
        spot=float(front["spot"]),
        basis=float(front["close"] - front["spot"]),
        open_interest=float(front["open_interest"]),
    )
