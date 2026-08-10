"""Download daily market prices from Yahoo Finance."""

from datetime import date

import pandas as pd
import yfinance as yf

from ml.config import DEFAULT_CONFIG, ModelConfig


class PriceDownloadError(RuntimeError):
    """Raised when price data cannot be downloaded."""


def download_prices(config: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Download adjusted daily prices using the model settings."""

    ticker = config.ticker.strip().upper()
    end_date = config.end_date or date.today().isoformat()

    try:
        prices = yf.download(
            tickers=ticker,
            start=config.start_date,
            end=end_date,
            interval=config.interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
    except Exception as error:
        raise PriceDownloadError(
            f"Could not download price data for {ticker}."
        ) from error

    if prices is None or prices.empty:
        raise PriceDownloadError(f"No price data was found for {ticker}.")

    return prices
