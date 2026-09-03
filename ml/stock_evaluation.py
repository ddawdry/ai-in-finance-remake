"""Run the same walk-forward evaluation across several stocks."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.data_cache import DEFAULT_CACHE_DIR, get_prices
from ml.dataset import build_model_dataset
from ml.split import DEFAULT_TRAIN_RATIO
from ml.walk_forward import DEFAULT_BLOCK_SIZE, run_walk_forward_evaluation


DEFAULT_STOCK_TICKERS = ("AAPL", "MSFT", "TSLA", "GOOGL", "AMZN")


class StockEvaluationError(ValueError):
    """Raised when a stock comparison cannot use the given tickers."""


def _read_tickers(tickers) -> tuple[str, ...]:
    if isinstance(tickers, str):
        raise StockEvaluationError("Tickers must be given as a list or tuple.")

    try:
        values = tuple(tickers)
    except TypeError as error:
        raise StockEvaluationError(
            "Tickers must be given as a list or tuple."
        ) from error

    if not values:
        raise StockEvaluationError("At least one ticker is required.")

    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in values):
        raise StockEvaluationError("Every ticker must be a non-empty name.")

    cleaned = tuple(ticker.strip().upper() for ticker in values)
    if len(set(cleaned)) != len(cleaned):
        raise StockEvaluationError("Tickers must not contain duplicates.")

    return cleaned


def evaluate_stocks(
    tickers=DEFAULT_STOCK_TICKERS,
    config: ModelConfig = DEFAULT_CONFIG,
    initial_train_ratio: float = DEFAULT_TRAIN_RATIO,
    block_size: int = DEFAULT_BLOCK_SIZE,
    refresh: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Evaluate each stock with the same model and date settings."""

    stock_tickers = _read_tickers(tickers)
    evaluations = {}
    rows = []

    for ticker in stock_tickers:
        stock_config = replace(config, ticker=ticker)
        prices = get_prices(stock_config, refresh=refresh, cache_dir=cache_dir)
        dataset = build_model_dataset(prices)
        evaluation = run_walk_forward_evaluation(
            dataset,
            initial_train_ratio=initial_train_ratio,
            block_size=block_size,
            config=stock_config,
        )
        evaluations[ticker] = evaluation
        rows.append(
            {
                "ticker": ticker,
                "asset_type": "stock",
                "calendar": "exchange_days",
                "dataset_rows": len(dataset),
                "prediction_rows": len(evaluation["predictions"]),
                **evaluation["metrics"],
            }
        )

    results = pd.DataFrame(rows).set_index("ticker")
    return {
        "settings": {
            "asset_type": "stock",
            "calendar": "exchange_days",
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
