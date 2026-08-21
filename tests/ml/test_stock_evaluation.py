from pathlib import Path

import pandas as pd
import pytest

from ml.config import ModelConfig
from ml.stock_evaluation import (
    DEFAULT_STOCK_TICKERS,
    StockEvaluationError,
    evaluate_stocks,
)


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {"prices": [], "walk_forward": []}

    def fake_get_prices(config, refresh, cache_dir):
        calls["prices"].append((config, refresh, cache_dir))
        return pd.DataFrame({"ticker": [config.ticker]})

    def fake_build_model_dataset(prices):
        return pd.DataFrame(
            {"target_up": [0, 1, 0, 1]},
            index=pd.bdate_range("2025-01-01", periods=4),
        )

    def fake_walk_forward(dataset, initial_train_ratio, block_size, config):
        calls["walk_forward"].append(
            (initial_train_ratio, block_size, config)
        )
        return {
            "predictions": pd.Series([0, 1]),
            "metrics": {
                "accuracy": 0.5,
                "precision": 0.6,
                "recall": 0.7,
                "f1": 0.65,
            },
        }

    monkeypatch.setattr("ml.stock_evaluation.get_prices", fake_get_prices)
    monkeypatch.setattr(
        "ml.stock_evaluation.build_model_dataset", fake_build_model_dataset
    )
    monkeypatch.setattr(
        "ml.stock_evaluation.run_walk_forward_evaluation", fake_walk_forward
    )
    return calls


def test_default_stock_list_is_small_and_fixed():
    assert DEFAULT_STOCK_TICKERS == ("AAPL", "MSFT", "GOOGL", "AMZN")


def test_every_stock_uses_the_same_settings(fake_pipeline):
    config = ModelConfig(start_date="2020-01-01", random_seed=7)

    result = evaluate_stocks(config=config)

    used_configs = [call[0] for call in fake_pipeline["prices"]]
    assert [config.ticker for config in used_configs] == list(
        DEFAULT_STOCK_TICKERS
    )
    assert all(config.start_date == "2020-01-01" for config in used_configs)
    assert all(config.interval == "1d" for config in used_configs)
    assert all(config.random_seed == 7 for config in used_configs)
    assert result["settings"]["initial_train_ratio"] == 0.8
    assert result["settings"]["block_size"] == 60


def test_results_table_contains_each_stock_and_metric(fake_pipeline):
    result = evaluate_stocks(tickers=["aapl", " msft "])

    assert result["results"].index.tolist() == ["AAPL", "MSFT"]
    assert result["results"].columns.tolist() == [
        "dataset_rows",
        "prediction_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    assert result["results"]["dataset_rows"].tolist() == [4, 4]
    assert result["results"]["prediction_rows"].tolist() == [2, 2]


def test_cache_and_walk_forward_options_are_passed_on(fake_pipeline):
    cache_dir = Path("local-test-cache")

    evaluate_stocks(
        tickers=["AAPL"],
        initial_train_ratio=0.7,
        block_size=25,
        refresh=True,
        cache_dir=cache_dir,
    )

    price_call = fake_pipeline["prices"][0]
    walk_call = fake_pipeline["walk_forward"][0]
    assert price_call[1:] == (True, cache_dir)
    assert walk_call[:2] == (0.7, 25)


def test_each_full_evaluation_is_kept(fake_pipeline):
    result = evaluate_stocks(tickers=["AAPL", "MSFT"])

    assert set(result["evaluations"]) == {"AAPL", "MSFT"}
    assert result["evaluations"]["AAPL"]["metrics"]["accuracy"] == 0.5


@pytest.mark.parametrize(
    "bad_tickers",
    [[], "AAPL", [""], [None], ["AAPL", "aapl"], None],
)
def test_bad_ticker_lists_are_rejected(bad_tickers):
    with pytest.raises(StockEvaluationError):
        evaluate_stocks(tickers=bad_tickers)
