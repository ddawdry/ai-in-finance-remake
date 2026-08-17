"""Create and score simple direction baselines."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.dataset import TARGET_COLUMN


class BaselineError(ValueError):
    """Raised when a baseline cannot use the given data."""


def _read_targets(data: pd.DataFrame, label: str) -> pd.Series:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise BaselineError(f"{label} data must not be empty.")

    if TARGET_COLUMN not in data.columns:
        raise BaselineError(f"{label} data must contain {TARGET_COLUMN}.")

    targets = data[TARGET_COLUMN]
    if targets.isna().any():
        raise BaselineError(f"{label} targets must not contain missing values.")

    if not set(targets.unique()).issubset({0, 1}):
        raise BaselineError(f"{label} targets must contain only 0 and 1.")

    return targets.astype("int64")


def _calculate_metrics(targets: pd.Series, predictions: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }


def run_majority_baseline(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    """Predict the most common training target for every test row."""

    training_targets = _read_targets(training_data, "Training")
    test_targets = _read_targets(test_data, "Test")

    up_count = int((training_targets == 1).sum())
    not_up_count = int((training_targets == 0).sum())
    majority_class = 1 if up_count > not_up_count else 0

    predictions = pd.Series(
        majority_class,
        index=test_data.index.copy(),
        dtype="int64",
        name="majority_prediction",
    )

    metrics = _calculate_metrics(test_targets, predictions)

    return {
        "majority_class": majority_class,
        "predictions": predictions,
        "metrics": metrics,
    }


def run_market_direction_baseline(test_data: pd.DataFrame) -> dict:
    """Predict that the next day follows the current day's direction."""

    test_targets = _read_targets(test_data, "Test")

    if "return_1d" not in test_data.columns:
        raise BaselineError("Test data must contain return_1d.")

    try:
        daily_returns = pd.to_numeric(test_data["return_1d"], errors="raise")
    except (TypeError, ValueError) as error:
        raise BaselineError("Test return_1d must contain numbers.") from error

    if daily_returns.isna().any():
        raise BaselineError("Test return_1d must not contain missing values.")

    if daily_returns.isin([float("inf"), float("-inf")]).any():
        raise BaselineError("Test return_1d must contain finite numbers.")

    predictions = (daily_returns > 0).astype("int64")
    predictions.name = "market_direction_prediction"

    return {
        "predictions": predictions,
        "metrics": _calculate_metrics(test_targets, predictions),
    }
