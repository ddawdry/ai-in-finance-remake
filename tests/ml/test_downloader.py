from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from ml.config import ModelConfig
from ml.downloader import PriceDownloadError, download_prices


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


@patch("ml.downloader.yf.download")
def test_download_uses_config_settings(mock_download):
    prices = sample_prices()
    mock_download.return_value = prices
    config = ModelConfig(
        ticker="aapl",
        start_date="2020-01-01",
        end_date="2025-01-01",
    )

    result = download_prices(config)

    assert result is prices
    mock_download.assert_called_once_with(
        tickers="AAPL",
        start="2020-01-01",
        end="2025-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )


@patch("ml.downloader.yf.download")
def test_default_end_date_excludes_the_current_day(mock_download):
    mock_download.return_value = sample_prices()

    download_prices(ModelConfig())

    assert mock_download.call_args.kwargs["end"] == date.today().isoformat()


@patch("ml.downloader.yf.download")
def test_empty_download_raises_clear_error(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(
        PriceDownloadError, match="No price data was found for AAPL"
    ):
        download_prices(ModelConfig())


@patch("ml.downloader.yf.download")
def test_missing_download_raises_clear_error(mock_download):
    mock_download.return_value = None

    with pytest.raises(
        PriceDownloadError, match="No price data was found for AAPL"
    ):
        download_prices(ModelConfig())


@patch("ml.downloader.yf.download")
def test_download_failure_raises_clear_error(mock_download):
    mock_download.side_effect = ConnectionError("Network unavailable")

    with pytest.raises(
        PriceDownloadError, match="Could not download price data for AAPL"
    ):
        download_prices(ModelConfig())
