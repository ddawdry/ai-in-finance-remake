"""Backtest direction predictions against buy and hold."""

from math import sqrt

import pandas as pd


class BacktestError(ValueError):
    """Raised when a backtest cannot use the given data."""


def calculate_performance_metrics(
    daily_returns: pd.Series,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Calculate a small set of return and risk measurements."""

    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        raise BacktestError("Daily returns must not be empty.")

    if (
        isinstance(trading_days_per_year, bool)
        or not isinstance(trading_days_per_year, int)
        or trading_days_per_year <= 0
    ):
        raise BacktestError("Trading days per year must be a positive whole number.")

    if (
        isinstance(risk_free_rate, bool)
        or not isinstance(risk_free_rate, (int, float))
        or risk_free_rate <= -1
    ):
        raise BacktestError("Risk-free rate must be a number greater than -1.")

    try:
        returns = pd.to_numeric(daily_returns, errors="raise").astype("float64")
    except (TypeError, ValueError) as error:
        raise BacktestError("Daily returns must contain numbers.") from error

    if returns.isna().any() or returns.isin([float("inf"), float("-inf")]).any():
        raise BacktestError("Daily returns must contain finite numbers.")

    if (returns < -1).any():
        raise BacktestError("Daily returns must not be lower than -1.")

    growth = (1 + returns).cumprod()
    total_return = float(growth.iloc[-1] - 1)
    if total_return <= -1:
        annualised_return = -1.0
    else:
        annualised_return = float(
            (1 + total_return) ** (trading_days_per_year / len(returns)) - 1
        )

    daily_volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualised_volatility = daily_volatility * sqrt(trading_days_per_year)

    growth_with_start = pd.concat(
        [pd.Series([1.0]), growth.reset_index(drop=True)], ignore_index=True
    )
    drawdowns = growth_with_start / growth_with_start.cummax() - 1
    maximum_drawdown = float(drawdowns.min())

    daily_risk_free_rate = risk_free_rate / trading_days_per_year
    excess_returns = returns - daily_risk_free_rate
    if daily_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(
            excess_returns.mean() / daily_volatility * sqrt(trading_days_per_year)
        )

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "maximum_drawdown": maximum_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


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
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
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
    strategy_metrics_before_costs = calculate_performance_metrics(
        daily["strategy_return"], trading_days_per_year, risk_free_rate
    )
    strategy_metrics_after_costs = calculate_performance_metrics(
        daily["strategy_return_after_costs"],
        trading_days_per_year,
        risk_free_rate,
    )
    buy_hold_metrics = calculate_performance_metrics(
        daily["buy_hold_return"], trading_days_per_year, risk_free_rate
    )

    return {
        "daily": daily,
        "trading_cost": float(trading_cost),
        "assumptions": {
            "trading_cost": float(trading_cost),
            "trading_days_per_year": trading_days_per_year,
            "risk_free_rate": float(risk_free_rate),
        },
        "strategy_metrics_before_costs": strategy_metrics_before_costs,
        "strategy_metrics_after_costs": strategy_metrics_after_costs,
        "buy_hold_metrics": buy_hold_metrics,
        "strategy_total_return": strategy_before_costs,
        "strategy_total_return_before_costs": strategy_before_costs,
        "strategy_total_return_after_costs": float(
            daily["strategy_growth_after_costs"].iloc[-1] - 1
        ),
        "buy_hold_total_return": float(daily["buy_hold_growth"].iloc[-1] - 1),
    }
