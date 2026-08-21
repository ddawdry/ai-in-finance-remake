"""Backtest direction predictions against buy and hold."""

import pandas as pd


class BacktestError(ValueError):
    """Raised when a backtest cannot use the given data."""


def _read_close_prices(prices: pd.DataFrame) -> pd.Series:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise BacktestError("Price data must not be empty.")

    if "Close" not in prices.columns:
        raise BacktestError("Price data must contain Close.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise BacktestError("Price data must use dates as its index.")

    if (
        prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise BacktestError("Price dates must be unique and sorted.")

    try:
        close = pd.to_numeric(prices["Close"], errors="raise")
    except (TypeError, ValueError) as error:
        raise BacktestError("Close must contain numbers.") from error

    if close.isna().any() or (close <= 0).any():
        raise BacktestError("Close must contain positive values.")

    return close


def _read_predictions(predictions: pd.Series) -> pd.Series:
    if not isinstance(predictions, pd.Series) or predictions.empty:
        raise BacktestError("Predictions must not be empty.")

    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise BacktestError("Predictions must use dates as their index.")

    if (
        predictions.index.has_duplicates
        or not predictions.index.is_monotonic_increasing
    ):
        raise BacktestError("Prediction dates must be unique and sorted.")

    if predictions.isna().any() or not set(predictions.unique()).issubset(
        {0, 1}
    ):
        raise BacktestError("Predictions must contain only 0 and 1.")

    return predictions.astype("int64")


def run_direction_backtest(
    prices: pd.DataFrame,
    predictions: pd.Series,
    trading_cost: float = 0.001,
) -> dict:
    """Compare an up-only direction strategy with buy and hold."""

    if (
        isinstance(trading_cost, bool)
        or not isinstance(trading_cost, (int, float))
        or not 0 <= trading_cost < 1
    ):
        raise BacktestError("Trading cost must be a number from 0 up to 1.")

    close = _read_close_prices(prices)
    signals = _read_predictions(predictions)

    missing_dates = signals.index.difference(close.index)
    if not missing_dates.empty:
        raise BacktestError("Every prediction date must exist in the price data.")

    next_day_returns = close.pct_change(fill_method=None).shift(-1).reindex(
        signals.index
    )
    if next_day_returns.isna().any():
        raise BacktestError("Every prediction needs a following price day.")

    daily = pd.DataFrame(index=signals.index.copy())
    daily.index.name = signals.index.name
    daily["prediction"] = signals
    daily["next_day_return"] = next_day_returns
    daily["strategy_return"] = signals * next_day_returns
    previous_position = signals.shift(1, fill_value=0)
    daily["position_changed"] = signals.ne(previous_position)
    daily["trading_cost"] = (
        daily["position_changed"].astype("int64") * trading_cost
    )
    daily["strategy_return_after_costs"] = (
        daily["strategy_return"] - daily["trading_cost"]
    )
    daily["buy_hold_return"] = next_day_returns
    daily["strategy_growth"] = (1 + daily["strategy_return"]).cumprod()
    daily["strategy_growth_after_costs"] = (
        1 + daily["strategy_return_after_costs"]
    ).cumprod()
    daily["buy_hold_growth"] = (1 + daily["buy_hold_return"]).cumprod()

    strategy_before_costs = float(daily["strategy_growth"].iloc[-1] - 1)
    return {
        "daily": daily,
        "trading_cost": float(trading_cost),
        "strategy_total_return": strategy_before_costs,
        "strategy_total_return_before_costs": strategy_before_costs,
        "strategy_total_return_after_costs": float(
            daily["strategy_growth_after_costs"].iloc[-1] - 1
        ),
        "buy_hold_total_return": float(daily["buy_hold_growth"].iloc[-1] - 1),
    }
