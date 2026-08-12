from unittest.mock import patch

import pandas as pd
import pytest

from ml.config import ModelConfig
from ml.data_cache import (
    PriceCacheError,
    get_cache_path,
    get_prices,
    load_prices,
    save_prices,
)


def sample_prices():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )


def test_cache_path_uses_ticker_and_interval(tmp_path):
    config = ModelConfig(ticker="aapl")

    path = get_cache_path(config, tmp_path)

    assert path == tmp_path / "AAPL_1d.csv"


def test_save_creates_folder_and_csv(tmp_path):
    cache_dir = tmp_path / "new" / "raw"

    path = save_prices(sample_prices(), cache_dir=cache_dir)

    assert path.is_file()
    assert path == cache_dir / "AAPL_1d.csv"


def test_saved_prices_can_be_loaded(tmp_path):
    original = sample_prices()
    save_prices(original, cache_dir=tmp_path)

    loaded = load_prices(cache_dir=tmp_path)

    pd.testing.assert_frame_equal(
        loaded, original, check_freq=False, check_names=False
    )
    assert loaded.index.name == "Date"


def test_save_does_not_change_original_dataframe(tmp_path):
    original = sample_prices()
    assert original.index.name is None

    save_prices(original, cache_dir=tmp_path)

    assert original.index.name is None


@pytest.mark.parametrize("prices", [None, pd.DataFrame()])
def test_empty_prices_cannot_be_saved(prices, tmp_path):
    with pytest.raises(PriceCacheError, match="Cannot save empty price data"):
        save_prices(prices, cache_dir=tmp_path)


def test_missing_cache_has_clear_error(tmp_path):
    with pytest.raises(PriceCacheError, match="No cached prices were found"):
        load_prices(cache_dir=tmp_path)


def test_invalid_cache_has_clear_error(tmp_path):
    cache_path = tmp_path / "AAPL_1d.csv"
    cache_path.write_text("Wrong,Columns\n1,2\n", encoding="utf-8")

    with pytest.raises(PriceCacheError, match="Could not load cached prices"):
        load_prices(cache_dir=tmp_path)


@patch("ml.data_cache.download_prices")
def test_get_prices_uses_existing_cache(mock_download, tmp_path):
    original = sample_prices()
    save_prices(original, cache_dir=tmp_path)

    result = get_prices(cache_dir=tmp_path)

    pd.testing.assert_frame_equal(
        result, original, check_freq=False, check_names=False
    )
    mock_download.assert_not_called()


@patch("ml.data_cache.download_prices")
def test_first_run_downloads_and_saves_prices(mock_download, tmp_path):
    prices = sample_prices()
    mock_download.return_value = prices

    result = get_prices(cache_dir=tmp_path)

    assert result is prices
    mock_download.assert_called_once()
    assert (tmp_path / "AAPL_1d.csv").is_file()


@patch("ml.data_cache.download_prices")
def test_refresh_downloads_and_replaces_cache(mock_download, tmp_path):
    old_prices = sample_prices()
    new_prices = sample_prices() * 2
    save_prices(old_prices, cache_dir=tmp_path)
    mock_download.return_value = new_prices

    result = get_prices(refresh=True, cache_dir=tmp_path)

    assert result is new_prices
    loaded = load_prices(cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        loaded, new_prices, check_freq=False, check_names=False
    )
