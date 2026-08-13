"""Check historical price data before it is used."""

import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


class PriceValidationError(ValueError):
    """Raised when historical price data is not safe to use."""


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a checked and cleaned copy of historical prices."""

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise PriceValidationError("Price data must not be empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in prices.columns
    ]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise PriceValidationError(
            f"Price data is missing required columns: {names}."
        )

    cleaned = prices.copy()

    try:
        cleaned.index = pd.to_datetime(cleaned.index, errors="raise")
    except (TypeError, ValueError) as error:
        raise PriceValidationError(
            "Price data index must contain valid dates."
        ) from error

    if cleaned.index.hasnans:
        raise PriceValidationError("Price data index contains a missing date.")

    if cleaned.index.duplicated().any():
        raise PriceValidationError("Price data contains duplicate dates.")

    for column in REQUIRED_COLUMNS:
        try:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise PriceValidationError(
                f"Column {column} must contain numbers."
            ) from error

    if cleaned[REQUIRED_COLUMNS].isna().any().any():
        raise PriceValidationError(
            "Price data contains missing values in required columns."
        )

    if (cleaned[PRICE_COLUMNS] <= 0).any().any():
        raise PriceValidationError("Price values must be greater than zero.")

    if (cleaned["Volume"] < 0).any():
        raise PriceValidationError("Volume must be zero or higher.")

    row_high = cleaned[["Open", "Close"]].max(axis=1)
    if (cleaned["High"] < row_high).any():
        raise PriceValidationError(
            "High must not be lower than the open, low, or close."
        )

    row_low = cleaned[["Open", "Close"]].min(axis=1)
    if (cleaned["Low"] > row_low).any():
        raise PriceValidationError(
            "Low must not be higher than the open, high, or close."
        )

    if (cleaned["High"] < cleaned["Low"]).any():
        raise PriceValidationError("High must not be lower than low.")

    cleaned = cleaned.sort_index()
    cleaned.index.name = "Date"
    return cleaned
