from pathlib import Path

import pandas as pd

from ml.config import ModelConfig
from ml.model import run_model


def test_run_model_uses_the_direction_pipeline(monkeypatch, tmp_path):
    calls = {}
    dataset = pd.DataFrame(
        {"target_up": [0, 1]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    evaluation = {
        "predictions": pd.Series([1], index=dataset.index[-1:]),
        "probabilities": pd.Series([0.7], index=dataset.index[-1:]),
        "metrics": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    }
    backtest = {"assumptions": {"trading_days_per_year": 252}}

    def fake_get_prices(config, refresh, cache_dir):
        calls["prices"] = (config, refresh, cache_dir)
        return pd.DataFrame({"Close": [100, 101]})

    def fake_build_model_dataset(prices):
        calls["dataset_input"] = prices
        return dataset

    def fake_walk_forward(model_dataset, config):
        calls["walk_forward"] = (model_dataset, config)
        return evaluation

    def fake_backtest(prices, predictions, trading_days_per_year):
        calls["backtest"] = (prices, predictions, trading_days_per_year)
        return backtest

    def fake_export(
        model_dataset, model_evaluation, ticker, asset_type, output_dir
    ):
        calls["export"] = (
            model_dataset,
            model_evaluation,
            ticker,
            asset_type,
            output_dir,
        )
        return {
            "predictions_path": output_dir / "direction_predictions.csv",
            "metrics_path": output_dir / "model_metrics.json",
            "model_results_path": output_dir / "model_results.json",
        }

    def fake_backtest_export(result, ticker, asset_type, output_dir):
        calls["backtest_export"] = (result, ticker, asset_type, output_dir)
        return output_dir / "backtest_results.json"

    monkeypatch.setattr("ml.model.get_prices", fake_get_prices)
    monkeypatch.setattr("ml.model.build_model_dataset", fake_build_model_dataset)
    monkeypatch.setattr(
        "ml.model.run_walk_forward_evaluation", fake_walk_forward
    )
    monkeypatch.setattr("ml.model.run_direction_backtest", fake_backtest)
    monkeypatch.setattr("ml.model.export_direction_results", fake_export)
    monkeypatch.setattr(
        "ml.model.export_backtest_results", fake_backtest_export
    )
    config = ModelConfig(ticker="AAPL")

    result = run_model(
        config=config,
        refresh=True,
        cache_dir=Path("cache"),
        output_dir=tmp_path,
    )

    assert calls["prices"] == (config, True, Path("cache"))
    assert calls["walk_forward"] == (dataset, config)
    assert calls["backtest"][1:] == (evaluation["predictions"], 252)
    assert calls["export"][2:] == ("AAPL", "stock", tmp_path)
    assert calls["backtest_export"] == (backtest, "AAPL", "stock", tmp_path)
    assert result["prediction_rows"] == 1
    assert result["metrics"]["accuracy"] == 1.0


def test_crypto_ticker_gets_crypto_label(monkeypatch, tmp_path):
    dataset = pd.DataFrame(
        {"target_up": [0, 1]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    evaluation = {
        "predictions": pd.Series([1], index=dataset.index[-1:]),
        "metrics": {"accuracy": 1.0},
    }
    saved = {}

    monkeypatch.setattr("ml.model.get_prices", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("ml.model.build_model_dataset", lambda prices: dataset)
    monkeypatch.setattr(
        "ml.model.run_walk_forward_evaluation",
        lambda model_dataset, config: evaluation,
    )
    monkeypatch.setattr(
        "ml.model.run_direction_backtest",
        lambda prices, predictions, trading_days_per_year: {},
    )

    def fake_export(*args, **kwargs):
        saved.update(kwargs)
        return {
            "predictions_path": tmp_path / "direction_predictions.csv",
            "metrics_path": tmp_path / "model_metrics.json",
            "model_results_path": tmp_path / "model_results.json",
        }

    monkeypatch.setattr("ml.model.export_direction_results", fake_export)
    monkeypatch.setattr(
        "ml.model.export_backtest_results",
        lambda *args, **kwargs: tmp_path / "backtest_results.json",
    )

    result = run_model(ModelConfig(ticker="BTC-USD"), output_dir=tmp_path)

    assert saved["asset_type"] == "crypto"
    assert result["asset_type"] == "crypto"
