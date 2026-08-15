"""Create model features from historical price data."""

import pandas as pd


RETURN_WINDOWS = (1, 3, 5)
MOVING_AVERAGE_WINDOWS = (5, 10, 20)
VOLATILITY_WINDOWS = (5, 20)


class FeatureError(ValueError):
    """Raised when features cannot be created from the given data."""


def _copy_with_numeric_close(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
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
    return result, close


def add_lagged_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add returns from one, three, and five trading days ago."""

    result, close = _copy_with_numeric_close(prices)

    for days in RETURN_WINDOWS:
        result[f"return_{days}d"] = close.pct_change(
            periods=days,
            fill_method=None,
        )

    return result


def add_moving_averages(prices: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages and close-to-average ratios."""

    result, close = _copy_with_numeric_close(prices)

    for days in MOVING_AVERAGE_WINDOWS:
        average_column = f"ma_{days}d"
        ratio_column = f"close_to_ma_{days}d"
        result[average_column] = close.rolling(
            window=days,
            min_periods=days,
        ).mean()
        result[ratio_column] = close / result[average_column]

    return result


def add_rolling_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility based on daily returns."""

    result, close = _copy_with_numeric_close(prices)
    daily_returns = close.pct_change(fill_method=None)

    for days in VOLATILITY_WINDOWS:
        result[f"volatility_{days}d"] = daily_returns.rolling(
            window=days,
            min_periods=days,
        ).std()

    return result


def add_volume_and_range_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add volume change, daily range, and open-to-close change."""

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise FeatureError("Price data must not be empty.")

    missing_columns = [
        column for column in required_columns if column not in prices.columns
    ]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise FeatureError(f"Price data is missing required columns: {names}.")

    result = prices.copy()

    for column in required_columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise FeatureError(f"{column} must contain numbers.") from error

    if result[required_columns].isna().any().any():
        raise FeatureError("Required price data must not contain missing values.")

    if (result[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise FeatureError("Price values must be greater than zero.")

    if (result["Volume"] < 0).any():
        raise FeatureError("Volume must be zero or higher.")

    previous_volume = result["Volume"].shift(1)
    result["volume_change_1d"] = (
        result["Volume"] - previous_volume
    ) / previous_volume
    result.loc[previous_volume == 0, "volume_change_1d"] = float("nan")

    result["daily_range"] = (
        result["High"] - result["Low"]
    ) / result["Low"]
    result["open_to_close"] = (
        result["Close"] - result["Open"]
    ) / result["Open"]

    return result


def add_direction_target(prices: pd.DataFrame) -> pd.DataFrame:
    """Add the next trading day's up-or-not target."""

    result, close = _copy_with_numeric_close(prices)
    next_close = close.shift(-1)
    target = (next_close > close).astype("Int64")
    target.loc[next_close.isna()] = pd.NA
    result["target_up"] = target
    return result
