"""Settings used by the direction model."""

from dataclasses import dataclass
from datetime import date


SUPPORTED_INTERVALS = {"1d"}


@dataclass(frozen=True)
class ModelConfig:
    """Store and check the main model settings."""

    ticker: str = "AAPL"
    start_date: str = "2016-01-01"
    end_date: str | None = None
    interval: str = "1d"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("Ticker must not be empty.")

        start = self._read_date(self.start_date, "Start date")

        if self.end_date is not None:
            end = self._read_date(self.end_date, "End date")
            if end <= start:
                raise ValueError("End date must be after the start date.")

        if self.interval not in SUPPORTED_INTERVALS:
            raise ValueError("Interval must be one of: 1d.")

        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise ValueError("Random seed must be a whole number.")

    @staticmethod
    def _read_date(value: str, label: str) -> date:
        if not isinstance(value, str):
            raise ValueError(f"{label} must use YYYY-MM-DD format.")

        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"{label} must use YYYY-MM-DD format."
            ) from error


DEFAULT_CONFIG = ModelConfig()
