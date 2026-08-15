import pandas as pd
import pytest

from ml.features import (
    FeatureError,
    add_lagged_returns,
    add_moving_averages,
    add_rolling_volatility,
    add_volume_and_range_features,
)


def sample_prices():
    return pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]},
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )


def moving_average_prices():
    return pd.DataFrame(
        {"Close": [float(price) for price in range(100, 120)]},
        index=pd.date_range("2025-01-01", periods=20, freq="D"),
    )


def steady_prices():
    closes = [100.0]
    for _ in range(20):
        closes.append(closes[-1] * 1.01)
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2025-01-01", periods=21, freq="D"),
    )


def jumpy_prices():
    closes = [100.0]
    for change in [0.10, -0.08, 0.12, -0.06, 0.09] * 4:
        closes.append(closes[-1] * (1 + change))
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2025-01-01", periods=21, freq="D"),
    )


def market_prices():
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0, 105.0],
            "High": [110.0, 120.0, 115.0],
            "Low": [90.0, 100.0, 100.0],
            "Close": [105.0, 100.0, 110.0],
            "Volume": [1000, 1500, 1200],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )


def test_adds_all_lagged_return_columns():
    result = add_lagged_returns(sample_prices())

    assert "return_1d" in result.columns
    assert "return_3d" in result.columns
    assert "return_5d" in result.columns


def test_one_day_returns_are_correct():
    result = add_lagged_returns(sample_prices())

    assert pd.isna(result.iloc[0]["return_1d"])
    assert result.iloc[1]["return_1d"] == pytest.approx(0.10)
    assert result.iloc[5]["return_1d"] == pytest.approx(0.10)


def test_three_day_returns_are_correct():
    result = add_lagged_returns(sample_prices())

    assert result["return_3d"].iloc[:3].isna().all()
    assert result.iloc[3]["return_3d"] == pytest.approx(0.331)


def test_five_day_returns_are_correct():
    result = add_lagged_returns(sample_prices())

    assert result["return_5d"].iloc[:5].isna().all()
    assert result.iloc[5]["return_5d"] == pytest.approx(0.61051)


def test_original_data_is_not_changed():
    original = sample_prices()
    original_before = original.copy(deep=True)

    add_lagged_returns(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_future_price_does_not_change_earlier_returns():
    original = sample_prices()
    changed = sample_prices()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 500.0

    original_result = add_lagged_returns(original)
    changed_result = add_lagged_returns(changed)

    pd.testing.assert_frame_equal(
        original_result.iloc[:-1],
        changed_result.iloc[:-1],
    )


def test_numeric_text_is_converted():
    prices = sample_prices().astype(str)

    result = add_lagged_returns(prices)

    assert result["Close"].dtype.kind in "fi"
    assert result.iloc[1]["return_1d"] == pytest.approx(0.10)


def test_empty_data_is_rejected():
    with pytest.raises(FeatureError, match="must not be empty"):
        add_lagged_returns(pd.DataFrame())


def test_missing_close_column_is_rejected():
    prices = pd.DataFrame({"Open": [100.0, 101.0]})

    with pytest.raises(FeatureError, match="must contain a Close column"):
        add_lagged_returns(prices)


def test_invalid_close_value_is_rejected():
    prices = pd.DataFrame({"Close": [100.0, "unknown"]})

    with pytest.raises(FeatureError, match="Close must contain numbers"):
        add_lagged_returns(prices)


def test_missing_close_value_is_rejected():
    prices = pd.DataFrame({"Close": [100.0, None]})

    with pytest.raises(FeatureError, match="must not contain missing values"):
        add_lagged_returns(prices)


@pytest.mark.parametrize("bad_close", [0, -1])
def test_non_positive_close_value_is_rejected(bad_close):
    prices = pd.DataFrame({"Close": [100.0, bad_close]})

    with pytest.raises(FeatureError, match="greater than zero"):
        add_lagged_returns(prices)


def test_adds_all_moving_average_columns():
    result = add_moving_averages(moving_average_prices())

    for days in (5, 10, 20):
        assert f"ma_{days}d" in result.columns
        assert f"close_to_ma_{days}d" in result.columns


def test_five_day_moving_average_and_ratio_are_correct():
    result = add_moving_averages(moving_average_prices())

    assert result["ma_5d"].iloc[:4].isna().all()
    assert result.iloc[4]["ma_5d"] == pytest.approx(102.0)
    assert result.iloc[4]["close_to_ma_5d"] == pytest.approx(104.0 / 102.0)


def test_ten_day_moving_average_and_ratio_are_correct():
    result = add_moving_averages(moving_average_prices())

    assert result["ma_10d"].iloc[:9].isna().all()
    assert result.iloc[9]["ma_10d"] == pytest.approx(104.5)
    assert result.iloc[9]["close_to_ma_10d"] == pytest.approx(109.0 / 104.5)


def test_twenty_day_moving_average_and_ratio_are_correct():
    result = add_moving_averages(moving_average_prices())

    assert result["ma_20d"].iloc[:19].isna().all()
    assert result.iloc[19]["ma_20d"] == pytest.approx(109.5)
    assert result.iloc[19]["close_to_ma_20d"] == pytest.approx(119.0 / 109.5)


def test_moving_averages_do_not_change_original_data():
    original = moving_average_prices()
    original_before = original.copy(deep=True)

    add_moving_averages(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_future_price_does_not_change_earlier_moving_averages():
    original = moving_average_prices()
    changed = moving_average_prices()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 500.0

    original_result = add_moving_averages(original)
    changed_result = add_moving_averages(changed)

    pd.testing.assert_frame_equal(
        original_result.iloc[:-1],
        changed_result.iloc[:-1],
    )


def test_adds_both_volatility_columns():
    result = add_rolling_volatility(steady_prices())

    assert "volatility_5d" in result.columns
    assert "volatility_20d" in result.columns


def test_volatility_waits_for_enough_daily_returns():
    result = add_rolling_volatility(steady_prices())

    assert result["volatility_5d"].iloc[:5].isna().all()
    assert result["volatility_20d"].iloc[:20].isna().all()
    assert pd.notna(result.iloc[5]["volatility_5d"])
    assert pd.notna(result.iloc[20]["volatility_20d"])


def test_steady_returns_have_zero_volatility():
    result = add_rolling_volatility(steady_prices())

    assert result.iloc[-1]["volatility_5d"] == pytest.approx(0.0, abs=1e-12)
    assert result.iloc[-1]["volatility_20d"] == pytest.approx(0.0, abs=1e-12)


def test_jumpy_prices_have_higher_volatility():
    steady_result = add_rolling_volatility(steady_prices())
    jumpy_result = add_rolling_volatility(jumpy_prices())

    assert (
        jumpy_result.iloc[-1]["volatility_5d"]
        > steady_result.iloc[-1]["volatility_5d"]
    )
    assert (
        jumpy_result.iloc[-1]["volatility_20d"]
        > steady_result.iloc[-1]["volatility_20d"]
    )


def test_five_day_volatility_is_correct():
    prices = jumpy_prices()
    expected_returns = pd.Series([0.10, -0.08, 0.12, -0.06, 0.09])

    result = add_rolling_volatility(prices)

    assert result.iloc[5]["volatility_5d"] == pytest.approx(
        expected_returns.std()
    )


def test_volatility_does_not_change_original_data():
    original = jumpy_prices()
    original_before = original.copy(deep=True)

    add_rolling_volatility(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_future_price_does_not_change_earlier_volatility():
    original = jumpy_prices()
    changed = jumpy_prices()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 500.0

    original_result = add_rolling_volatility(original)
    changed_result = add_rolling_volatility(changed)

    pd.testing.assert_frame_equal(
        original_result.iloc[:-1],
        changed_result.iloc[:-1],
    )


def test_adds_volume_and_range_columns():
    result = add_volume_and_range_features(market_prices())

    assert "volume_change_1d" in result.columns
    assert "daily_range" in result.columns
    assert "open_to_close" in result.columns


def test_volume_change_is_correct():
    result = add_volume_and_range_features(market_prices())

    assert pd.isna(result.iloc[0]["volume_change_1d"])
    assert result.iloc[1]["volume_change_1d"] == pytest.approx(0.5)
    assert result.iloc[2]["volume_change_1d"] == pytest.approx(-0.2)


def test_daily_range_is_correct():
    result = add_volume_and_range_features(market_prices())

    assert result.iloc[0]["daily_range"] == pytest.approx((110 - 90) / 90)


def test_open_to_close_change_is_correct():
    result = add_volume_and_range_features(market_prices())

    assert result.iloc[0]["open_to_close"] == pytest.approx(0.05)
    assert result.iloc[1]["open_to_close"] == pytest.approx(-10 / 110)


def test_zero_previous_volume_does_not_create_infinity():
    prices = market_prices()
    prices.loc[prices.index[0], "Volume"] = 0

    result = add_volume_and_range_features(prices)

    assert pd.isna(result.iloc[1]["volume_change_1d"])


def test_zero_current_volume_is_allowed():
    prices = market_prices()
    prices.loc[prices.index[1], "Volume"] = 0

    result = add_volume_and_range_features(prices)

    assert result.iloc[1]["volume_change_1d"] == pytest.approx(-1.0)


def test_missing_volume_has_clear_error():
    prices = market_prices().drop(columns="Volume")

    with pytest.raises(FeatureError, match="missing required columns: Volume"):
        add_volume_and_range_features(prices)


def test_volume_and_range_features_do_not_change_original_data():
    original = market_prices()
    original_before = original.copy(deep=True)

    add_volume_and_range_features(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_future_row_does_not_change_earlier_volume_and_range_features():
    original = market_prices()
    changed = market_prices()
    changed.iloc[-1] = [200.0, 220.0, 190.0, 210.0, 5000]

    original_result = add_volume_and_range_features(original)
    changed_result = add_volume_and_range_features(changed)

    pd.testing.assert_frame_equal(
        original_result.iloc[:-1],
        changed_result.iloc[:-1],
    )


def test_volume_and_range_numeric_text_is_converted():
    prices = market_prices().astype(str)

    result = add_volume_and_range_features(prices)

    assert result["Volume"].dtype.kind in "fi"
    assert result.iloc[1]["volume_change_1d"] == pytest.approx(0.5)
