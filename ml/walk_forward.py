"""Evaluate the Random Forest by moving forward through time."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.classifier import train_random_forest
from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.dataset import TARGET_COLUMN
from ml.split import DEFAULT_TRAIN_RATIO, split_dataset


DEFAULT_BLOCK_SIZE = 60


class WalkForwardError(ValueError):
    """Raised when walk-forward evaluation settings are not valid."""


def run_walk_forward_evaluation(
    dataset: pd.DataFrame,
    initial_train_ratio: float = DEFAULT_TRAIN_RATIO,
    block_size: int = DEFAULT_BLOCK_SIZE,
    config: ModelConfig = DEFAULT_CONFIG,
) -> dict:
    """Train on past rows and predict each following block in order."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise WalkForwardError("Block size must be a positive whole number.")

    initial_training, remaining_data = split_dataset(
        dataset, train_ratio=initial_train_ratio
    )
    first_test_position = len(initial_training)

    prediction_parts = []
    probability_parts = []
    folds = []

    for fold_number, test_start in enumerate(
        range(first_test_position, len(dataset), block_size), start=1
    ):
        test_end = min(test_start + block_size, len(dataset))
        training_data = dataset.iloc[:test_start].copy()
        test_data = dataset.iloc[test_start:test_end].copy()
        result = train_random_forest(training_data, test_data, config)

        prediction_parts.append(result["predictions"])
        probability_parts.append(result["probabilities"])
        folds.append(
            {
                "fold": fold_number,
                "training_rows": len(training_data),
                "training_start": training_data.index.min(),
                "training_end": training_data.index.max(),
                "test_rows": len(test_data),
                "test_start": test_data.index.min(),
                "test_end": test_data.index.max(),
            }
        )

    predictions = pd.concat(prediction_parts).astype("int64")
    predictions.name = "walk_forward_prediction"
    probabilities = pd.concat(probability_parts).astype("float64")
    probabilities.name = "walk_forward_up_probability"
    targets = remaining_data[TARGET_COLUMN].astype("int64")

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }

    return {
        "initial_training_rows": len(initial_training),
        "block_size": block_size,
        "folds": folds,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }
