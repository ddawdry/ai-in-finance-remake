"""Save and load downloaded prices on the local computer."""

from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.downloader import download_prices


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class PriceCacheError(RuntimeError):
    """Raised when cached price data cannot be saved or loaded."""


def get_cache_path(
    config: ModelConfig = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the local CSV path for a ticker and interval."""

    ticker = config.ticker.strip().upper()
    return Path(cache_dir) / f"{ticker}_{config.interval}.csv"


def save_prices(
    prices: pd.DataFrame,
    config: ModelConfig = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Save prices as CSV and return the saved file path."""

    if prices is None or prices.empty:
        raise PriceCacheError("Cannot save empty price data.")

    cache_path = get_cache_path(config, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = prices.copy()
    data_to_save.index.name = "Date"

    try:
        data_to_save.to_csv(cache_path)
    except (OSError, ValueError) as error:
        raise PriceCacheError(
            f"Could not save cached prices to {cache_path}."
        ) from error

    return cache_path


def load_prices(
    config: ModelConfig = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load prices from the matching local CSV file."""

    cache_path = get_cache_path(config, cache_dir)

    if not cache_path.is_file():
        raise PriceCacheError(f"No cached prices were found at {cache_path}.")

    try:
        prices = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    except (OSError, ValueError, KeyError) as error:
        raise PriceCacheError(
            f"Could not load cached prices from {cache_path}."
        ) from error

    if prices.empty:
        raise PriceCacheError(f"Cached prices at {cache_path} are empty.")

    return prices


def get_prices(
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load cached prices or download and save a fresh copy."""

    cache_path = get_cache_path(config, cache_dir)

    if cache_path.is_file() and not refresh:
        return load_prices(config, cache_dir)

    prices = download_prices(config)
    save_prices(prices, config, cache_dir)
    return prices
