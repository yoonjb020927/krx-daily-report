"""ETF signals and an event backtest that checks whether each signal has worked.

Timing rule (no look-ahead): a signal uses data up to day t's close,
the trade enters at day t+1's open and exits at day t+hold's close.

A signal is judged on *excess* return: its trades versus buying every ETF in
the liquid universe on the same day with the same timing. In a rising market
almost anything makes money; only beating that baseline shows an edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import config

Panel = dict[str, pd.DataFrame]          # field -> (date x code) table
SignalFn = Callable[[Panel], pd.DataFrame]
WINDOW = 20


def build_panel(etf_history: pd.DataFrame) -> Panel:
    frame = etf_history.drop_duplicates(["date", "code"], keep="last")
    # drop holiday snapshots (every close blank) so rolling windows count trading days only
    traded = frame.groupby("date")["close"].transform(lambda s: s.notna().any())
    frame = frame[traded]
    panel = {
        field: frame.pivot(index="date", columns="code", values=field).sort_index()
        for field in ("open", "close", "value")
    }
    # a zero price means no trade that day
    return {k: (v.where(v > 0) if k != "value" else v.fillna(0)) for k, v in panel.items()}


def liquid_universe(panel: Panel, size: int) -> pd.DataFrame:
    """Per-date top-N ETFs by trailing average traded value (avoids hindsight picks)."""
    avg_value = panel["value"].rolling(WINDOW, min_periods=WINDOW).mean()
    return avg_value.rank(axis=1, ascending=False) <= size


def value_surge(panel: Panel) -> pd.DataFrame:
    value, close = panel["value"], panel["close"]
    prior_avg = value.rolling(WINDOW, min_periods=WINDOW).mean().shift(1)
    return (value > 2 * prior_avg) & (close > close.shift(1))


def ma_golden_cross(panel: Panel) -> pd.DataFrame:
    close = panel["close"]
    fast, slow = close.rolling(5).mean(), close.rolling(WINDOW).mean()
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def breakout_20d(panel: Panel) -> pd.DataFrame:
    close = panel["close"]
    return close > close.rolling(WINDOW, min_periods=WINDOW).max().shift(1)


SIGNALS: dict[str, tuple[str, SignalFn]] = {
    "value_surge": ("거래대금 급증(20일 평균 2배) + 상승", value_surge),
    "ma_golden_cross": ("5일선이 20일선 상향 돌파", ma_golden_cross),
    "breakout_20d": ("20일 신고가 돌파", breakout_20d),
}


@dataclass(frozen=True)
class SignalStats:
    key: str
    label: str
    trades: int
    signal_days: int
    avg_return: float      # per trade, after costs
    avg_excess: float      # per signal day, vs. same-day universe average
    t_stat: float          # of daily excess, overlap-adjusted
    oos_excess: float      # avg excess in the most recent (held-out) part
    is_valid: bool


def forward_returns(panel: Panel, hold_days: int) -> pd.DataFrame:
    """Gross return of entering at t+1 open and exiting at t+hold close, indexed by t."""
    return panel["close"].shift(-hold_days) / panel["open"].shift(-1) - 1


def trade_returns(panel: Panel, fired: pd.DataFrame, hold_days: int, cost: float) -> pd.Series:
    returns = forward_returns(panel, hold_days) - cost
    picked = returns.where(_as_bool(fired))
    return picked.stack().dropna()


def _as_bool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.fillna(False).astype(bool)


def _t_stat(series: pd.Series, hold_days: int) -> float:
    """t of the mean; divided by sqrt(hold) because 5-day holds started on
    consecutive days overlap and are not independent."""
    n = series.size
    if n < 2 or series.std() == 0 or math.isnan(series.std()):
        return 0.0
    return float(series.mean() / (series.std() / math.sqrt(n)) / math.sqrt(hold_days))


def _signal_stats(key: str, label: str, fired: pd.DataFrame, panel: Panel, excess: pd.DataFrame,
                  oos_start: str, settings: config.SignalSettings) -> SignalStats:
    net = trade_returns(panel, fired, settings.hold_days, settings.round_trip_cost)
    daily_excess = excess.where(fired).mean(axis=1).dropna()
    oos = daily_excess[daily_excess.index >= oos_start]
    avg_excess = float(daily_excess.mean()) if daily_excess.size else 0.0
    oos_excess = float(oos.mean()) if oos.size else float("nan")
    t_stat = _t_stat(daily_excess, settings.hold_days)
    valid = (
        int(net.size) >= settings.min_trades
        and daily_excess.size >= settings.min_signal_days
        and avg_excess > 0
        and t_stat >= settings.min_t_stat
        and oos_excess > 0
        and float(net.mean()) > 0
    )
    return SignalStats(key, label, int(net.size), int(daily_excess.size),
                       float(net.mean()) if net.size else 0.0, avg_excess, t_stat, oos_excess, valid)


def evaluate(panel: Panel, settings: config.SignalSettings = config.SIGNALS) -> list[SignalStats]:
    universe = liquid_universe(panel, settings.universe_size)
    fwd = forward_returns(panel, settings.hold_days)
    baseline = fwd.where(universe).mean(axis=1)          # buy the whole universe that day
    excess = fwd.sub(baseline, axis=0)
    dates = panel["close"].index
    oos_start = dates[int(len(dates) * (1 - settings.oos_fraction))] if len(dates) else ""
    return [
        _signal_stats(key, label, _as_bool(fn(panel) & universe), panel, excess, oos_start, settings)
        for key, (label, fn) in SIGNALS.items()
    ]


def todays_candidates(
    panel: Panel, etf_today: pd.DataFrame, stats: list[SignalStats],
    settings: config.SignalSettings = config.SIGNALS,
) -> pd.DataFrame:
    """ETFs that fired a *validated* signal on the latest date."""
    universe = liquid_universe(panel, settings.universe_size)
    last = panel["close"].index[-1]
    rows = []
    for stat in (s for s in stats if s.is_valid):
        fired = _as_bool(SIGNALS[stat.key][1](panel) & universe).loc[last]
        for code in fired[fired].index:
            rows.append({"code": code, "signal": stat.label, "avg_excess": stat.avg_excess,
                         "t_stat": stat.t_stat})
    columns = ["code", "name", "close", "signal", "avg_excess", "t_stat"]
    if not rows:
        return pd.DataFrame(columns=columns)
    names = etf_today.set_index("code")[["name", "close"]]
    found = pd.DataFrame(rows).join(names, on="code")
    return found[columns].sort_values("avg_excess", ascending=False).reset_index(drop=True)
