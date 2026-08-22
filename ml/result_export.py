"""Export clean direction predictions and model metrics."""

import json
import re
from pathlib import Path

import pandas as pd

from ml.dataset import TARGET_COLUMN


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"
PREDICTIONS_FILE = "direction_predictions.csv"
METRICS_FILE = "model_metrics.json"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


class ResultExportError(ValueError):
    """Raised when model results cannot be safely exported."""


def _read_name(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or re.fullmatch(r"[A-Za-z0-9._-]+", value.strip()) is None
    ):
        raise ResultExportError(
            f"{label} must use letters, numbers, dots, dashes, or underscores."
        )
    return value.strip()


def export_direction_results(
    dataset: pd.DataFrame,
    evaluation: dict,
    ticker: str,
    asset_type: str,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict:
    """Save direction rows as CSV and summary metrics as JSON."""

    safe_ticker = _read_name(ticker, "Ticker").upper()
    safe_asset_type = _read_name(asset_type, "Asset type").lower()
    if safe_asset_type not in {"stock", "crypto"}:
        raise ResultExportError("Asset type must be stock or crypto.")

    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ResultExportError("Dataset must not be empty.")
    if TARGET_COLUMN not in dataset.columns:
        raise ResultExportError(f"Dataset must contain {TARGET_COLUMN}.")

    if not isinstance(evaluation, dict):
        raise ResultExportError("Evaluation must be a result dictionary.")
    try:
        predictions = evaluation["predictions"]
        probabilities = evaluation["probabilities"]
        raw_metrics = evaluation["metrics"]
    except KeyError as error:
        raise ResultExportError(
            "Evaluation must contain predictions, probabilities, and metrics."
        ) from error

    if not isinstance(predictions, pd.Series) or predictions.empty:
        raise ResultExportError("Predictions must not be empty.")
    if not isinstance(probabilities, pd.Series):
        raise ResultExportError("Probabilities must be a pandas Series.")
    if not predictions.index.equals(probabilities.index):
        raise ResultExportError("Prediction and probability dates must match.")
    if not predictions.index.isin(dataset.index).all():
        raise ResultExportError("Every prediction date must exist in the dataset.")
    if not set(predictions.unique()).issubset({0, 1}):
        raise ResultExportError("Predictions must contain only 0 and 1.")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ResultExportError("Probabilities must be between 0 and 1.")

    try:
        metrics = {name: float(raw_metrics[name]) for name in METRIC_NAMES}
    except (KeyError, TypeError, ValueError) as error:
        raise ResultExportError(
            "Metrics must contain numeric accuracy, precision, recall, and f1."
        ) from error

    targets = dataset.loc[predictions.index, TARGET_COLUMN].astype("int64")
    direction_names = {0: "down", 1: "up"}
    rows = pd.DataFrame(
        {
            "date": predictions.index.strftime("%Y-%m-%d"),
            "ticker": safe_ticker,
            "asset_type": safe_asset_type,
            "actual_direction": targets.map(direction_names).to_numpy(),
            "predicted_direction": predictions.map(direction_names).to_numpy(),
            "up_probability": probabilities.astype("float64").to_numpy(),
        }
    )

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    predictions_path = folder / PREDICTIONS_FILE
    metrics_path = folder / METRICS_FILE
    rows.to_csv(predictions_path, index=False)

    metric_output = {
        "ticker": safe_ticker,
        "asset_type": safe_asset_type,
        "model": "random_forest_walk_forward",
        "prediction_rows": len(rows),
        "start_date": rows["date"].iloc[0],
        "end_date": rows["date"].iloc[-1],
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metric_output, file, indent=2)

    return {
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
    }
