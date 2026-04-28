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
    LOG_DIR: str = "logs"


SETTINGS = Settings()


def et_now() -> datetime:
    return datetime.now(ET)


def et_today() -> date:
    return et_now().date()
