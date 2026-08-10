import pytest

from ml.config import DEFAULT_CONFIG, ModelConfig


def test_default_config_matches_model_scope():
    assert DEFAULT_CONFIG.ticker == "AAPL"
    assert DEFAULT_CONFIG.start_date == "2016-01-01"
    assert DEFAULT_CONFIG.end_date is None
    assert DEFAULT_CONFIG.interval == "1d"
    assert DEFAULT_CONFIG.random_seed == 42


@pytest.mark.parametrize("ticker", ["", "   ", None])
def test_empty_or_missing_ticker_is_rejected(ticker):
    with pytest.raises(ValueError, match="Ticker must not be empty"):
        ModelConfig(ticker=ticker)


@pytest.mark.parametrize("start_date", ["01-01-2016", "not-a-date", None])
def test_bad_start_date_is_rejected(start_date):
    with pytest.raises(ValueError, match="Start date must use YYYY-MM-DD"):
        ModelConfig(start_date=start_date)


def test_bad_end_date_is_rejected():
    with pytest.raises(ValueError, match="End date must use YYYY-MM-DD"):
        ModelConfig(end_date="31-12-2025")


@pytest.mark.parametrize("end_date", ["2015-12-31", "2016-01-01"])
def test_end_date_must_be_after_start_date(end_date):
    with pytest.raises(ValueError, match="End date must be after"):
        ModelConfig(end_date=end_date)


def test_unsupported_interval_is_rejected():
    with pytest.raises(ValueError, match="Interval must be one of"):
        ModelConfig(interval="1h")


@pytest.mark.parametrize("random_seed", [4.2, "42", True, None])
def test_random_seed_must_be_a_whole_number(random_seed):
    with pytest.raises(ValueError, match="Random seed must be a whole number"):
        ModelConfig(random_seed=random_seed)
