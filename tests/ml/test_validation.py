import pandas as pd
import pytest

from ml.validation import PriceValidationError, validate_prices


def sample_prices():
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 101.0],
            "Close": [102.0, 103.0],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )


def test_valid_prices_pass_validation():
    result = validate_prices(sample_prices())

    assert len(result) == 2
    assert result.index.name == "Date"


def test_validation_does_not_change_original_data():
    original = sample_prices().astype(str)
    original_before = original.copy(deep=True)

    result = validate_prices(original)

    pd.testing.assert_frame_equal(original, original_before)
    assert result.iloc[0]["Open"] == 100.0


def test_dates_are_sorted_oldest_first():
    prices = sample_prices().sort_index(ascending=False)

    result = validate_prices(prices)

    assert result.index.is_monotonic_increasing


def test_missing_columns_are_rejected():
    prices = sample_prices().drop(columns=["Volume", "Low"])

    with pytest.raises(
        PriceValidationError,
        match="missing required columns: Low, Volume",
    ):
        validate_prices(prices)


def test_duplicate_dates_are_rejected():
    prices = sample_prices()
    prices.index = pd.to_datetime(["2025-01-02", "2025-01-02"])

    with pytest.raises(PriceValidationError, match="duplicate dates"):
        validate_prices(prices)


def test_missing_date_is_rejected():
    prices = sample_prices()
    prices.index = pd.DatetimeIndex(["2025-01-02", None])

    with pytest.raises(PriceValidationError, match="missing date"):
        validate_prices(prices)


def test_invalid_date_is_rejected():
    prices = sample_prices()
    prices.index = ["2025-01-02", "not-a-date"]

    with pytest.raises(PriceValidationError, match="valid dates"):
        validate_prices(prices)


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
def test_missing_required_values_are_rejected(column):
    prices = sample_prices()
    prices.loc[prices.index[0], column] = None

    with pytest.raises(PriceValidationError, match="missing values"):
        validate_prices(prices)


def test_numeric_text_is_converted():
    prices = sample_prices().astype(str)

    result = validate_prices(prices)

    assert result["Close"].dtype.kind in "fi"
    assert result.iloc[0]["Close"] == 102.0


def test_invalid_numeric_text_is_rejected():
    prices = sample_prices()
    prices["Close"] = prices["Close"].astype(object)
    prices.loc[prices.index[0], "Close"] = "unknown"

    with pytest.raises(
        PriceValidationError, match="Column Close must contain numbers"
    ):
        validate_prices(prices)


@pytest.mark.parametrize("value", [0, -1])
def test_zero_or_negative_prices_are_rejected(value):
    prices = sample_prices()
    prices.loc[prices.index[0], "Open"] = value

    with pytest.raises(PriceValidationError, match="greater than zero"):
        validate_prices(prices)


def test_negative_volume_is_rejected():
    prices = sample_prices()
    prices.loc[prices.index[0], "Volume"] = -1

    with pytest.raises(PriceValidationError, match="zero or higher"):
        validate_prices(prices)


def test_zero_volume_is_allowed():
    prices = sample_prices()
    prices.loc[prices.index[0], "Volume"] = 0

    result = validate_prices(prices)

    assert result.iloc[0]["Volume"] == 0


def test_high_below_other_prices_is_rejected():
    prices = sample_prices()
    prices.loc[prices.index[0], "High"] = 98.0

    with pytest.raises(PriceValidationError, match="High must not be lower"):
        validate_prices(prices)


def test_low_above_other_prices_is_rejected():
    prices = sample_prices()
    prices.loc[prices.index[0], "Low"] = 105.0

    with pytest.raises(PriceValidationError, match="Low must not be higher"):
        validate_prices(prices)


def test_extra_columns_are_kept():
    prices = sample_prices()
    prices["Source"] = "sample"

    result = validate_prices(prices)

    assert result["Source"].tolist() == ["sample", "sample"]
