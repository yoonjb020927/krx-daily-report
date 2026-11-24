import math

import pandas as pd
import pytest

from krxreport import indicators, parsing


@pytest.mark.parametrize("raw, expected", [("1,234", 1234.0), ("-", math.nan), ("", math.nan),
                                           (None, math.nan), ("12.5", 12.5), (7, 7.0), ("abc", math.nan)])
def test_to_number_handles_krx_strings(raw, expected):
    result = parsing.to_number(raw)
    assert (math.isnan(result) and math.isnan(expected)) or result == expected


def test_normalize_options_parses_strike_expiry_and_right(source):
    frame = parsing.normalize("options", source.fetch("options", source.days[0]))
    row = frame[frame["name"] == "코스피200 P 202612   340.0 (정규)"].iloc[0]
    assert (row["right"], row["expiry"], row["strike"], row["session"]) == ("P", "202612", 340.0, "regular")
    night = frame[frame["name"] == "코스피200 P 202612   340.0 (야간)"].iloc[0]
    assert night["session"] == "night"
    assert frame["volume"].dtype.kind == "f"


@pytest.mark.parametrize("name, strike", [
    ("코스피200 C 202612 1,002.5 (정규)", 1002.5),
    ("코스피200 P 202612   752.5 (야간)", 752.5),
    ("코스피200 C 202612 12,345.0 (정규)", 12345.0),
])
def test_option_strike_with_thousands_comma(name, strike):
    row = {"BAS_DD": "20261120", "PROD_NM": "코스피200 옵션", "RGHT_TP_NM": "CALL", "ISU_CD": "x",
           "ISU_NM": name, "TDD_CLSPRC": "", "ACC_TRDVOL": "0", "ACC_OPNINT_QTY": "0"}
    assert parsing.normalize("options", [row])["strike"].iloc[0] == strike


def test_expiry_day_uses_next_series_with_open_interest(source):
    options = parsing.normalize("options", source.fetch("options", source.days[0]))
    # expiry day: the expiring 202612 series is listed with zero open interest
    expiring = options["expiry"] == "202612"
    on_expiry = options.assign(open_interest=options["open_interest"].where(~expiring, 0.0))
    summary = indicators.summarize_options(on_expiry)
    assert summary.front_expiry == "202701"
    assert summary.max_pain > 320            # not the lowest strike


def test_large_max_pain_gap_is_flagged_not_blocked(source):
    from dataclasses import replace

    from krxreport import report
    day = source.days[0]
    opt = indicators.summarize_options(parsing.normalize("options", source.fetch("options", day)))
    fut = indicators.summarize_futures(parsing.normalize("futures", source.fetch("futures", day)))
    assert indicators.max_pain_gap(opt, fut) < 0.01
    assert "⚠" not in report._max_pain_note(opt, fut)
    assert "⚠ 괴리" in report._max_pain_note(replace(opt, max_pain=250.0), fut)


def test_duplicate_strikes_are_rejected(source):
    options = parsing.normalize("options", source.fetch("options", source.days[0]))
    regular = options[options["session"] == "regular"]
    with pytest.raises(indicators.DataQualityError, match="duplicate"):
        indicators.summarize_options(pd.concat([regular, regular.head(5)]))


def test_normalize_raises_clear_error_when_field_missing():
    with pytest.raises(parsing.SchemaError, match="TDD_CLSPRC"):
        parsing.normalize("etf", [{"BAS_DD": "20261123", "ISU_CD": "1"}])


def test_normalize_empty_rows_returns_empty_frame():
    assert parsing.normalize("etf", []).empty


def test_pcr_ignores_mini_options_that_would_inflate_it(source):
    options = parsing.normalize("options", source.fetch("options", source.days[0]))
    naive = options.loc[options["right"] == "P", "volume"].sum() / options.loc[options["right"] == "C", "volume"].sum()
    summary = indicators.summarize_options(options)
    assert naive > 10            # mixing products gives a nonsense ratio
    assert 0.5 < summary.pcr_volume < 2


def test_max_pain_hand_example():
    frame = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0, 90.0, 100.0, 110.0],
        "right": ["C", "C", "C", "P", "P", "P"],
        # holders' payout = 100*|S-100| -> smallest at 100
        "open_interest": [0, 100, 0, 0, 100, 0],
    })
    assert indicators.max_pain(frame) == 100.0


def test_max_pain_matches_brute_force(source):
    options = indicators.kospi200_options(parsing.normalize("options", source.fetch("options", source.days[0])))
    front = options[options["expiry"] == "202612"]

    def pain(settle: float) -> float:
        calls = front[front.right == "C"]
        puts = front[front.right == "P"]
        return float(((settle - calls.strike).clip(lower=0) * calls.open_interest).sum()
                     + ((puts.strike - settle).clip(lower=0) * puts.open_interest).sum())

    expected = min(front.strike.unique(), key=pain)
    assert indicators.max_pain(front) == expected
    assert 345 <= expected <= 355


def test_summarize_options_rejects_implausible_pcr(source):
    options = parsing.normalize("options", source.fetch("options", source.days[0]))
    skewed = options.assign(volume=options["volume"].where(options["right"] == "P", 1.0))
    with pytest.raises(indicators.DataQualityError, match="PCR"):
        indicators.summarize_options(skewed)


def test_summarize_options_rejects_wrong_product_name(source, monkeypatch):
    from krxreport import config
    monkeypatch.setattr(config, "OPTIONS_PRODUCT", "없는상품")
    options = parsing.normalize("options", source.fetch("options", source.days[0]))
    with pytest.raises(indicators.DataQualityError, match="rows"):
        indicators.summarize_options(options)


def test_futures_picks_front_month_regular_contract(source):
    futures = parsing.normalize("futures", source.fetch("futures", source.days[0]))
    summary = indicators.summarize_futures(futures)
    assert summary.name == "코스피200 F 202612 (주간)"   # not the night row, not the spread
    assert summary.basis == pytest.approx(1.2)
