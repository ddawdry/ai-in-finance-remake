from pathlib import Path

import pytest

from ml.config import DEFAULT_CONFIG
from ml.run_assets import DEFAULT_ASSET_TICKERS, clean_tickers, run_assets


def test_default_list_has_stocks_and_crypto():
    assert "AAPL" in DEFAULT_ASSET_TICKERS
    assert "MSFT" in DEFAULT_ASSET_TICKERS
    assert "TSLA" in DEFAULT_ASSET_TICKERS
    assert "BTC-USD" in DEFAULT_ASSET_TICKERS
    assert "ETH-USD" in DEFAULT_ASSET_TICKERS


def test_tickers_are_cleaned_and_duplicates_removed():
    assert clean_tickers([" aapl ", "BTC-USD", "AAPL"]) == (
        "AAPL",
        "BTC-USD",
    )


@pytest.mark.parametrize("ticker", ["", "../AAPL", ".", "AAPL/USD"])
def test_unsafe_tickers_are_rejected(ticker):
    with pytest.raises(ValueError, match="invalid"):
        clean_tickers([ticker])


def test_each_asset_uses_its_own_result_folder(monkeypatch, tmp_path):
    calls = []

    def fake_run_model(config, refresh, output_dir):
        calls.append((config.ticker, refresh, output_dir))
        return {"ticker": config.ticker}

    monkeypatch.setattr("ml.run_assets.run_model", fake_run_model)

    results = run_assets(
        ["AAPL", "BTC-USD"],
        config=DEFAULT_CONFIG,
        refresh=True,
        output_dir=tmp_path,
    )

    assert calls == [
        ("AAPL", True, Path(tmp_path) / "AAPL"),
        ("BTC-USD", True, Path(tmp_path) / "BTC-USD"),
    ]
    assert [result["ticker"] for result in results] == ["AAPL", "BTC-USD"]
