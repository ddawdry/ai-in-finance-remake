"""Create model features from historical price data."""

import pandas as pd


RETURN_WINDOWS = (1, 3, 5)


class FeatureError(ValueError):
    """Raised when features cannot be created from the given data."""


def add_lagged_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add returns from one, three, and five trading days ago."""

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise FeatureError("Price data must not be empty.")

    if "Close" not in prices.columns:
        raise FeatureError("Price data must contain a Close column.")

    result = prices.copy()

    try:
        close = pd.to_numeric(result["Close"], errors="raise")
    except (TypeError, ValueError) as error:
        raise FeatureError("Close must contain numbers.") from error

    if close.isna().any():
        raise FeatureError("Close must not contain missing values.")

    if (close <= 0).any():
        raise FeatureError("Close values must be greater than zero.")

    result["Close"] = close

    for days in RETURN_WINDOWS:
        result[f"return_{days}d"] = close.pct_change(
            periods=days,
            fill_method=None,
        )

    return result
