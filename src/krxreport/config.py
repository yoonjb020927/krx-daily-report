"""Settings in one place.

KRX field and product names below were checked against real responses
(probe on 20261120). If KRX changes them, run
`python -m krxreport probe --date YYYYMMDD` and update here.
"""
from dataclasses import dataclass

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
AUTH_HEADER = "AUTH_KEY"
RESPONSE_ROWS_KEY = "OutBlock_1"

ENDPOINTS = {
    "etf": "etp/etf_bydd_trd",
    "futures": "drv/fut_bydd_trd",
    "options": "drv/opt_bydd_trd",
}

# raw KRX field -> our column name
ETF_FIELDS = {
    "BAS_DD": "date",
    "ISU_CD": "code",
    "ISU_NM": "name",
    "TDD_OPNPRC": "open",
    "TDD_HGPRC": "high",
    "TDD_LWPRC": "low",
    "TDD_CLSPRC": "close",
    "ACC_TRDVOL": "volume",
    "ACC_TRDVAL": "value",
    "NAV": "nav",
}

FUTURES_FIELDS = {
    "BAS_DD": "date",
    "PROD_NM": "product",
    "ISU_CD": "code",
    "ISU_NM": "name",
    "TDD_CLSPRC": "close",
    "SPOT_PRC": "spot",
    "ACC_TRDVOL": "volume",
    "ACC_OPNINT_QTY": "open_interest",
}

OPTIONS_FIELDS = {
    "BAS_DD": "date",
    "PROD_NM": "product",
    "RGHT_TP_NM": "right",
    "ISU_CD": "code",
    "ISU_NM": "name",
    "TDD_CLSPRC": "close",
    "ACC_TRDVOL": "volume",
    "ACC_OPNINT_QTY": "open_interest",
}

FIELDS_BY_KIND = {"etf": ETF_FIELDS, "futures": FUTURES_FIELDS, "options": OPTIONS_FIELDS}

# Only the regular KOSPI200 contracts. Mini / weekly / KOSDAQ150 products are
# listed in the same response; mixing them in distorts PCR badly.
FUTURES_PRODUCT = "코스피200 선물"
OPTIONS_PRODUCT = "코스피200 옵션"


@dataclass(frozen=True)
class SignalSettings:
    universe_size: int = 100          # most-traded ETFs by 20-day avg value
    hold_days: int = 5
    round_trip_cost: float = 0.002    # fees + slippage, as a fraction
    min_trades: int = 30
    min_signal_days: int = 30         # distinct days the signal fired
    min_t_stat: float = 2.0           # excess return must be statistically clear
    oos_fraction: float = 0.3         # most recent 30% must also beat the baseline


@dataclass(frozen=True)
class SanityLimits:
    pcr_min: float = 0.05
    pcr_max: float = 20.0             # real volume PCR reached ~10 in the June 2026 rally
    max_pain_warn_gap: float = 0.2    # flag (not block) when |max pain - futures| / futures exceeds this
    min_etf_rows: int = 100
    min_option_rows: int = 20


SIGNALS = SignalSettings()
LIMITS = SanityLimits()
REQUEST_TIMEOUT_SEC = 15
REQUEST_RETRIES = 3
BACKFILL_SLEEP_SEC = 0.3
LOOKBACK_DAYS_FOR_LATEST = 7
