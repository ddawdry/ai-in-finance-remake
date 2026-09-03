"""Run the model for each asset used by the dashboard."""

import argparse
import re
from dataclasses import replace
from pathlib import Path

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.crypto_evaluation import DEFAULT_CRYPTO_TICKERS
from ml.model import run_model
from ml.result_export import DEFAULT_RESULTS_DIR
from ml.stock_evaluation import DEFAULT_STOCK_TICKERS


DEFAULT_ASSET_TICKERS = DEFAULT_STOCK_TICKERS + DEFAULT_CRYPTO_TICKERS
SAFE_TICKER = re.compile(r"[A-Za-z0-9._-]+")


def clean_tickers(tickers) -> tuple[str, ...]:
    """Return safe, unique ticker names in their given order."""

    cleaned = []
    for value in tickers:
        ticker = str(value).strip().upper()
        if (
            not ticker
            or SAFE_TICKER.fullmatch(ticker) is None
            or ticker in {".", ".."}
        ):
            raise ValueError(f"Ticker is invalid: {value}")
        if ticker not in cleaned:
            cleaned.append(ticker)

    if not cleaned:
        raise ValueError("At least one ticker is required.")
    return tuple(cleaned)


def run_assets(
    tickers=DEFAULT_ASSET_TICKERS,
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[dict]:
    """Build separate dashboard result files for several assets."""

    results = []
    for ticker in clean_tickers(tickers):
        print(f"Running {ticker}...")
        asset_config = replace(config, ticker=ticker)
        result = run_model(
            config=asset_config,
            refresh=refresh,
            output_dir=Path(output_dir) / ticker,
        )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dashboard results for stocks and crypto."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_ASSET_TICKERS,
        help="Yahoo Finance tickers. Defaults to the dashboard asset list.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh prices instead of using cached files.",
    )
    args = parser.parse_args()

    results = run_assets(args.tickers, refresh=args.refresh)
    print(f"Finished {len(results)} assets.")


if __name__ == "__main__":
    main()
