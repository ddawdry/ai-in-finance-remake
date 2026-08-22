"""Run the direction model and export its clean results."""

from pathlib import Path

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.data_cache import DEFAULT_CACHE_DIR, get_prices
from ml.dataset import build_model_dataset
from ml.result_export import DEFAULT_RESULTS_DIR, export_direction_results
from ml.walk_forward import run_walk_forward_evaluation


def run_model(
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict:
    """Build walk-forward direction results for one ticker."""

    prices = get_prices(config, refresh=refresh, cache_dir=cache_dir)
    dataset = build_model_dataset(prices)
    evaluation = run_walk_forward_evaluation(dataset, config=config)
    asset_type = "crypto" if config.ticker.upper().endswith("-USD") else "stock"
    paths = export_direction_results(
        dataset,
        evaluation,
        ticker=config.ticker,
        asset_type=asset_type,
        output_dir=output_dir,
    )

    return {
        "ticker": config.ticker.upper(),
        "asset_type": asset_type,
        "dataset_rows": len(dataset),
        "prediction_rows": len(evaluation["predictions"]),
        "metrics": evaluation["metrics"],
        **paths,
    }


if __name__ == "__main__":
    result = run_model()
    print(f"Exported {result['prediction_rows']} direction rows.")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Metrics: {result['metrics_path']}")
