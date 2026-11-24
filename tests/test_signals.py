import numpy as np
import pandas as pd
import pytest

from krxreport import config, signals


def _panel(close: list[float], opens: list[float] | None = None) -> signals.Panel:
    idx = [f"2026{m:04d}" for m in range(101, 101 + len(close))]
    frame = lambda v: pd.DataFrame({"A": v}, index=idx)  # noqa: E731
    return {"close": frame(close), "open": frame(opens or close), "value": frame([1.0] * len(close))}


def test_trade_enters_next_open_and_exits_after_hold_days():
    panel = _panel(close=[100, 100, 100, 120, 130], opens=[100, 100, 110, 115, 125])
    fired = pd.DataFrame({"A": [False, True, False, False, False]}, index=panel["close"].index)
    rets = signals.trade_returns(panel, fired, hold_days=2, cost=0.0)
    # signal on day 1 -> enter day 2 open (110) -> exit day 3 close (120)
    assert rets.tolist() == [pytest.approx(120 / 110 - 1)]


def test_signal_without_enough_future_data_is_dropped():
    panel = _panel(close=[100, 101, 102])
    fired = pd.DataFrame({"A": [False, False, True]}, index=panel["close"].index)
    assert signals.trade_returns(panel, fired, hold_days=2, cost=0.0).empty


def test_breakout_uses_only_past_highs():
    close = [100.0] * 20 + [101.0]
    fired = signals.breakout_20d(_panel(close))
    assert fired["A"].iloc[-1]
    assert not fired["A"].iloc[:-1].any()


def test_liquid_universe_ranks_per_date():
    idx = [str(i) for i in range(25)]
    value = pd.DataFrame({"big": [100.0] * 25, "small": [1.0] * 25}, index=idx)
    panel = {"value": value, "close": value, "open": value}
    universe = signals.liquid_universe(panel, size=1)
    assert universe["big"].iloc[-1] and not universe["small"].iloc[-1]
    assert not universe.iloc[0].any()   # not enough history yet


def test_evaluate_marks_signal_invalid_when_too_few_trades():
    panel = _panel(close=list(np.linspace(100, 110, 30)))
    settings = config.SignalSettings(universe_size=5, min_trades=1000)
    stats = signals.evaluate(panel, settings)
    assert {s.key for s in stats} == set(signals.SIGNALS)
    assert not any(s.is_valid for s in stats)


def _bull_market(n_days: int = 300, n_etfs: int = 20) -> signals.Panel:
    """Every ETF drifts up ~0.3%/day: buying anything makes money."""
    rng = np.random.default_rng(1)
    idx = [f"{i:08d}" for i in range(n_days)]
    cols = [f"E{j}" for j in range(n_etfs)]
    close = 100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, (n_days, n_etfs)), axis=0))
    frame = lambda v: pd.DataFrame(v, index=idx, columns=cols)  # noqa: E731
    return {"close": frame(close), "open": frame(close), "value": frame(np.ones_like(close))}


def _only_signal(monkeypatch, fn):
    monkeypatch.setattr(signals, "SIGNALS", {"test": ("테스트 시그널", fn)})


def test_signal_that_just_rides_a_bull_market_is_rejected(monkeypatch):
    panel = _bull_market()
    _only_signal(monkeypatch, lambda p: p["close"].notna())      # always "buy"
    [stat] = signals.evaluate(panel, config.SignalSettings(universe_size=20))
    assert stat.avg_return > 0.005                 # looks great on raw return...
    assert abs(stat.avg_excess) < 1e-9             # ...but it IS the baseline
    assert not stat.is_valid


def test_signal_with_real_edge_is_accepted(monkeypatch):
    panel = _bull_market()
    fwd = signals.forward_returns(panel, config.SIGNALS.hold_days)
    oracle = fwd.gt(fwd.median(axis=1), axis=0)                # picks the better half
    _only_signal(monkeypatch, lambda p: oracle)
    [stat] = signals.evaluate(panel, config.SignalSettings(universe_size=20))
    assert stat.avg_excess > 0 and stat.oos_excess > 0
    assert stat.t_stat >= 2
    assert stat.is_valid


def test_signal_that_stops_working_recently_is_rejected(monkeypatch):
    panel = _bull_market()
    fwd = signals.forward_returns(panel, config.SIGNALS.hold_days)
    good = fwd.gt(fwd.median(axis=1), axis=0)
    cutoff = int(len(fwd) * 0.75)
    decayed = pd.concat([good.iloc[:cutoff], ~good.iloc[cutoff:]])  # edge flips in the held-out part
    _only_signal(monkeypatch, lambda p: decayed)
    [stat] = signals.evaluate(panel, config.SignalSettings(universe_size=20))
    assert stat.oos_excess < 0
    assert not stat.is_valid


def test_todays_candidates_only_from_valid_signals():
    close = [100.0] * 25 + [110.0]
    panel = _panel(close)
    panel["value"] = pd.DataFrame({"A": [1.0] * 26}, index=panel["close"].index)
    today = pd.DataFrame({"code": ["A"], "name": ["테스트"], "close": [110.0]})
    valid = signals.SignalStats("breakout_20d", "20일 신고가 돌파", 50, 40, 0.01, 0.004, 2.5, 0.003, True)
    invalid = signals.SignalStats("value_surge", "x", 50, 40, -0.01, -0.002, -1.0, -0.001, False)
    picks = signals.todays_candidates(panel, today, [valid, invalid], config.SignalSettings(universe_size=5))
    assert picks["signal"].tolist() == ["20일 신고가 돌파"]
    assert picks["name"].tolist() == ["테스트"]
