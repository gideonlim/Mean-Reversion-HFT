from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Settings:
    # Symbols to trade. Each gets its own independent lag-1 mean-reversion signal.
    SYMBOLS: tuple[str, ...] = ("SPY",)
    # Optional per-symbol portfolio weights as ((symbol, weight), ...).
    # Empty -> equal weight (1/N each). Sum may be < 1 (rest stays in cash);
    # sum > 1 raises ValueError (would imply leverage).
    SYMBOL_WEIGHTS: tuple[tuple[str, float], ...] = ()

    POSITION_FRACTION: float = 0.95  # Total portfolio gross exposure cap.
    IN_SAMPLE_SPLIT: float = 0.75
    LOOKBACK_YEARS: int = 5
    ANNUALIZATION_N: int = 252
    LONG_ONLY: bool = False
    STARTING_CAPITAL: float = 50_000.0  # Used by report.py for return/P&L baseline.
    MIN_DAYS_FOR_ANNUALIZATION: int = 30  # Below this, report shows N/A for annualized stats.
    LOG_DIR: str = "logs"
    REPORT_DIR: str = "report"  # Daily PDF/CSV/chart outputs saved here.

    def get_weights(self) -> dict[str, float]:
        """Return the per-symbol weight map.

        Empty SYMBOL_WEIGHTS -> equal weight 1/N. Otherwise the explicit dict,
        with any symbol in SYMBOLS but missing from the dict assigned weight 0
        (no allocation).

        Raises ValueError on invalid configurations.
        """
        if not self.SYMBOLS:
            raise ValueError("SYMBOLS must be non-empty")

        if not self.SYMBOL_WEIGHTS:
            n = len(self.SYMBOLS)
            return {s: 1.0 / n for s in self.SYMBOLS}

        weights = dict(self.SYMBOL_WEIGHTS)
        symbols_set = set(self.SYMBOLS)

        unknown = set(weights) - symbols_set
        if unknown:
            raise ValueError(
                f"SYMBOL_WEIGHTS contains symbols not in SYMBOLS: {sorted(unknown)}"
            )
        for s, w in weights.items():
            if w < 0:
                raise ValueError(f"Weight for {s!r} must be non-negative, got {w}")

        total = sum(weights.values())
        # Allow tiny float drift; cap at 1 + 1e-9
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"SYMBOL_WEIGHTS sum to {total:.6f}, must be <= 1.0 (rest stays as cash)"
            )

        # Symbols in SYMBOLS but missing from weights -> weight 0
        for s in self.SYMBOLS:
            weights.setdefault(s, 0.0)
        return weights


SETTINGS = Settings()
# Validate at import time so a misconfigured deployment fails loudly.
SETTINGS.get_weights()


def et_now() -> datetime:
    return datetime.now(ET)


def et_today() -> date:
    return et_now().date()
