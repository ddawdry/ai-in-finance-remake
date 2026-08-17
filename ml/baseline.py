"""Create and score a simple majority-class baseline."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.dataset import TARGET_COLUMN


class BaselineError(ValueError):
    """Raised when the baseline cannot use the given targets."""


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

    metrics = {
        "accuracy": float(accuracy_score(test_targets, predictions)),
        "precision": float(
            precision_score(test_targets, predictions, zero_division=0)
        ),
        "recall": float(recall_score(test_targets, predictions, zero_division=0)),
        "f1": float(f1_score(test_targets, predictions, zero_division=0)),
    }

    return {
        "majority_class": majority_class,
        "predictions": predictions,
        "metrics": metrics,
    }
