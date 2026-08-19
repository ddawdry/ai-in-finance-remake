import pandas as pd
import pytest

from ml.backtest import BacktestError, run_direction_backtest


def price_data(closes):
    return pd.DataFrame(
        {"Close": closes},
        index=pd.bdate_range("2025-01-01", periods=len(closes), name="Date"),
    )


def direction_predictions(values):
    return pd.Series(
        values,
        index=pd.bdate_range("2025-01-01", periods=len(values), name="Date"),
        name="random_forest_prediction",
    )


def test_up_prediction_uses_the_next_day_return():
    result = run_direction_backtest(
        price_data([100, 110, 99]), direction_predictions([1, 1])
    )

    assert result["daily"]["next_day_return"].tolist() == pytest.approx(
        [0.10, -0.10]
    )
    assert result["daily"]["strategy_return"].tolist() == pytest.approx(
        [0.10, -0.10]
    )


def test_down_prediction_stays_out_of_the_market():
    result = run_direction_backtest(
        price_data([100, 110, 99]), direction_predictions([0, 0])
    )

    assert result["daily"]["strategy_return"].tolist() == [0.0, 0.0]
    assert result["strategy_total_return"] == 0.0


def test_buy_and_hold_uses_every_next_day_return():
    result = run_direction_backtest(
        price_data([100, 110, 99]), direction_predictions([0, 1])
    )

    assert result["daily"]["buy_hold_return"].tolist() == pytest.approx(
        [0.10, -0.10]
    )
    assert result["buy_hold_total_return"] == pytest.approx(-0.01)


def test_returns_are_compounded():
    result = run_direction_backtest(
        price_data([100, 110, 121]), direction_predictions([1, 1])
    )

    assert result["strategy_total_return"] == pytest.approx(0.21)
    assert result["daily"]["strategy_growth"].tolist() == pytest.approx(
        [1.10, 1.21]
    )


def test_original_data_is_not_changed():
    prices = price_data([100, 110, 99])
    predictions = direction_predictions([1, 0])
    prices_before = prices.copy(deep=True)
    predictions_before = predictions.copy(deep=True)

    run_direction_backtest(prices, predictions)

    pd.testing.assert_frame_equal(prices, prices_before)
    pd.testing.assert_series_equal(predictions, predictions_before)


@pytest.mark.parametrize("values", [[1, 2], [1, None], ["1", "0"]])
def test_bad_predictions_are_rejected(values):
    with pytest.raises(BacktestError, match="only 0 and 1"):
        run_direction_backtest(price_data([100, 110, 99]), direction_predictions(values))


def test_prediction_date_must_exist_in_prices():
    predictions = direction_predictions([1, 0])
    predictions.index = pd.bdate_range("2026-01-01", periods=2)

    with pytest.raises(BacktestError, match="must exist"):
        run_direction_backtest(price_data([100, 110, 99]), predictions)


def test_last_price_date_cannot_have_a_prediction():
    prices = price_data([100, 110, 99])
    predictions = pd.Series(
        [1], index=prices.index[-1:], dtype="int64"
    )

    with pytest.raises(BacktestError, match="following price day"):
        run_direction_backtest(prices, predictions)


def test_inputs_must_not_be_empty():
    with pytest.raises(BacktestError, match="Price data must not be empty"):
        run_direction_backtest(pd.DataFrame(), direction_predictions([1]))
