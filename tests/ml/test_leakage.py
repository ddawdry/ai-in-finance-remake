import pandas as pd
import pytest

from ml.backtest import run_direction_backtest
from ml.classifier import train_random_forest
from ml.dataset import FEATURE_COLUMNS, build_model_dataset
from ml.features import add_direction_target
from ml.split import split_dataset


def price_data(rows=80):
    closes = [100 + day + (day % 4) for day in range(rows)]
    return pd.DataFrame(
        {
            "Open": [close * 0.99 for close in closes],
            "High": [close * 1.02 for close in closes],
            "Low": [close * 0.98 for close in closes],
            "Close": closes,
            "Volume": [1000 + day * 20 for day in range(rows)],
        },
        index=pd.bdate_range("2024-01-01", periods=rows, name="Date"),
    )


def test_changing_a_future_price_does_not_change_older_features():
    original_prices = price_data()
    changed_prices = original_prices.copy(deep=True)
    future_date = changed_prices.index[-1]
    changed_prices.loc[future_date, ["Open", "High", "Low", "Close"]] = [
        50,
        51,
        49,
        50,
    ]

    original = build_model_dataset(original_prices)
    changed = build_model_dataset(changed_prices)
    older_dates = original.index[original.index < future_date]

    pd.testing.assert_frame_equal(
        original.loc[older_dates, FEATURE_COLUMNS],
        changed.loc[older_dates, FEATURE_COLUMNS],
    )


def test_next_close_only_changes_the_target_not_older_features():
    original_prices = price_data()
    changed_prices = original_prices.copy(deep=True)
    final_date = changed_prices.index[-1]
    prediction_date = changed_prices.index[-2]
    changed_prices.loc[final_date, ["Open", "High", "Low", "Close"]] = [
        50,
        51,
        49,
        50,
    ]

    original = build_model_dataset(original_prices)
    changed = build_model_dataset(changed_prices)

    pd.testing.assert_series_equal(
        original.loc[prediction_date, FEATURE_COLUMNS],
        changed.loc[prediction_date, FEATURE_COLUMNS],
    )
    assert original.loc[prediction_date, "target_up"] == 1
    assert changed.loc[prediction_date, "target_up"] == 0


def test_target_uses_the_following_close():
    prices = pd.DataFrame(
        {"Close": [100, 90, 120]},
        index=pd.bdate_range("2025-01-01", periods=3),
    )

    result = add_direction_target(prices)

    assert result["target_up"].iloc[0] == 0
    assert result["target_up"].iloc[1] == 1
    assert pd.isna(result["target_up"].iloc[2])


def test_model_predictions_only_use_the_later_test_rows():
    dataset = build_model_dataset(price_data())
    training_data, test_data = split_dataset(dataset)

    result = train_random_forest(training_data, test_data)

    assert training_data.index.max() < test_data.index.min()
    assert result["predictions"].index.equals(test_data.index)
    assert result["probabilities"].index.equals(test_data.index)


def test_backtest_uses_the_return_after_the_prediction_date():
    dates = pd.bdate_range("2025-01-01", periods=3, name="Date")
    prices = pd.DataFrame({"Close": [50, 100, 110]}, index=dates)
    predictions = pd.Series([1], index=dates[1:2], dtype="int64")

    result = run_direction_backtest(prices, predictions)

    assert result["daily"]["next_day_return"].iloc[0] == pytest.approx(0.10)
    assert result["daily"]["strategy_return"].iloc[0] == pytest.approx(0.10)
