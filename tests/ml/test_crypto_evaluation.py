from pathlib import Path

import pandas as pd
import pytest

from ml.config import ModelConfig
from ml.crypto_evaluation import (
    DEFAULT_CRYPTO_TICKERS,
    CryptoEvaluationError,
    evaluate_crypto,
)


@pytest.fixture
def fake_crypto_pipeline(monkeypatch):
    calls = {"prices": [], "walk_forward": []}

    def fake_get_prices(config, refresh, cache_dir):
        calls["prices"].append((config, refresh, cache_dir))
        return pd.DataFrame(
            {"ticker": [config.ticker] * 4},
            index=pd.date_range("2025-01-03", periods=4, freq="D"),
        )

    def fake_build_model_dataset(prices):
        return pd.DataFrame(
            {"target_up": [0, 1, 0, 1]},
            index=prices.index,
        )

    def fake_walk_forward(dataset, initial_train_ratio, block_size, config):
        calls["walk_forward"].append(
            (dataset.index.copy(), initial_train_ratio, block_size, config)
        )
        return {
            "predictions": pd.Series([0, 1], index=dataset.index[-2:]),
            "metrics": {
                "accuracy": 0.5,
                "precision": 0.6,
                "recall": 0.7,
                "f1": 0.65,
            },
        }

    monkeypatch.setattr("ml.crypto_evaluation.get_prices", fake_get_prices)
    monkeypatch.setattr(
        "ml.crypto_evaluation.build_model_dataset", fake_build_model_dataset
    )
    monkeypatch.setattr(
        "ml.crypto_evaluation.run_walk_forward_evaluation", fake_walk_forward
    )
    return calls


def test_default_crypto_list_matches_the_current_project():
    assert DEFAULT_CRYPTO_TICKERS == (
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "ADA-USD",
        "DOGE-USD",
    )


def test_every_crypto_uses_the_same_model_settings(fake_crypto_pipeline):
    config = ModelConfig(start_date="2020-01-01", random_seed=7)

    result = evaluate_crypto(config=config)

    used_configs = [call[0] for call in fake_crypto_pipeline["prices"]]
    assert [config.ticker for config in used_configs] == list(
        DEFAULT_CRYPTO_TICKERS
    )
    assert all(config.start_date == "2020-01-01" for config in used_configs)
    assert all(config.interval == "1d" for config in used_configs)
    assert all(config.random_seed == 7 for config in used_configs)
    assert result["settings"]["initial_train_ratio"] == 0.8
    assert result["settings"]["block_size"] == 60


def test_crypto_results_are_clearly_labelled(fake_crypto_pipeline):
    result = evaluate_crypto(tickers=["btc-usd", " eth-usd "])

    assert result["results"].index.tolist() == ["BTC-USD", "ETH-USD"]
    assert result["results"]["asset_type"].tolist() == ["crypto", "crypto"]
    assert result["results"]["calendar"].tolist() == [
        "seven_days",
        "seven_days",
    ]
    assert result["settings"]["calendar_days_per_year"] == 365


def test_weekend_dates_are_kept_in_the_pipeline(fake_crypto_pipeline):
    evaluate_crypto(tickers=["BTC-USD"])

    dates = fake_crypto_pipeline["walk_forward"][0][0]
    assert dates[1].day_name() == "Saturday"
    assert dates[2].day_name() == "Sunday"


def test_cache_and_walk_forward_options_are_passed_on(fake_crypto_pipeline):
    cache_dir = Path("local-crypto-cache")

    evaluate_crypto(
        tickers=["BTC-USD"],
        initial_train_ratio=0.7,
        block_size=25,
        refresh=True,
        cache_dir=cache_dir,
    )

    price_call = fake_crypto_pipeline["prices"][0]
    walk_call = fake_crypto_pipeline["walk_forward"][0]
    assert price_call[1:] == (True, cache_dir)
    assert walk_call[1:3] == (0.7, 25)


def test_each_full_crypto_evaluation_is_kept(fake_crypto_pipeline):
    result = evaluate_crypto(tickers=["BTC-USD", "ETH-USD"])

    assert set(result["evaluations"]) == {"BTC-USD", "ETH-USD"}
    assert result["evaluations"]["BTC-USD"]["metrics"]["accuracy"] == 0.5


@pytest.mark.parametrize(
    "bad_tickers",
    [[], "BTC-USD", [""], [None], ["BTC-USD", "btc-usd"], None],
)
def test_bad_crypto_ticker_lists_are_rejected(bad_tickers):
    with pytest.raises(CryptoEvaluationError):
        evaluate_crypto(tickers=bad_tickers)
