"""Run the walk-forward evaluation across cryptocurrencies."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.data_cache import DEFAULT_CACHE_DIR, get_prices
from ml.dataset import build_model_dataset
from ml.split import DEFAULT_TRAIN_RATIO
from ml.stock_evaluation import StockEvaluationError, _read_tickers
from ml.walk_forward import DEFAULT_BLOCK_SIZE, run_walk_forward_evaluation


DEFAULT_CRYPTO_TICKERS = (
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "DOGE-USD",
)


class CryptoEvaluationError(ValueError):
    """Raised when a crypto comparison cannot use the given tickers."""


def _read_crypto_tickers(tickers) -> tuple[str, ...]:
    try:
        return _read_tickers(tickers)
    except StockEvaluationError as error:
        raise CryptoEvaluationError(str(error)) from error


def evaluate_crypto(
    tickers=DEFAULT_CRYPTO_TICKERS,
    config: ModelConfig = DEFAULT_CONFIG,
    initial_train_ratio: float = DEFAULT_TRAIN_RATIO,
    block_size: int = DEFAULT_BLOCK_SIZE,
    refresh: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Evaluate each cryptocurrency without removing weekend dates."""

    crypto_tickers = _read_crypto_tickers(tickers)
    evaluations = {}
    rows = []

    for ticker in crypto_tickers:
        crypto_config = replace(config, ticker=ticker)
        prices = get_prices(crypto_config, refresh=refresh, cache_dir=cache_dir)
        dataset = build_model_dataset(prices)
        evaluation = run_walk_forward_evaluation(
            dataset,
            initial_train_ratio=initial_train_ratio,
            block_size=block_size,
            config=crypto_config,
        )
        evaluations[ticker] = evaluation
        rows.append(
            {
                "ticker": ticker,
                "asset_type": "crypto",
                "calendar": "seven_days",
                "dataset_rows": len(dataset),
                "prediction_rows": len(evaluation["predictions"]),
                **evaluation["metrics"],
            }
        )

    results = pd.DataFrame(rows).set_index("ticker")
    return {
        "settings": {
            "asset_type": "crypto",
            "calendar": "seven_days",
            "calendar_days_per_year": 365,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "interval": config.interval,
            "random_seed": config.random_seed,
            "initial_train_ratio": initial_train_ratio,
            "block_size": block_size,
        },
        "results": results,
        "evaluations": evaluations,
    }
