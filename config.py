from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Settings:
    SYMBOL: str = "SPY"
    POSITION_FRACTION: float = 0.95
    IN_SAMPLE_SPLIT: float = 0.75
    LOOKBACK_YEARS: int = 5
    ANNUALIZATION_N: int = 252
    LONG_ONLY: bool = False
    STARTING_CAPITAL: float = 50_000.0  # Used by report.py for return/P&L baseline
    MIN_DAYS_FOR_ANNUALIZATION: int = 30  # Below this, report shows N/A for annualized stats
    LOG_DIR: str = "logs"


SETTINGS = Settings()


def et_now() -> datetime:
    return datetime.now(ET)


def et_today() -> date:
    return et_now().date()
