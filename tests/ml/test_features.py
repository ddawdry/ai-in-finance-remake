import pandas as pd
import pytest

from ml.features import FeatureError, add_lagged_returns, add_moving_averages


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
