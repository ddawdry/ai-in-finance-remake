"""Compare direction models using the same test data."""

import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.baseline import run_majority_baseline, run_market_direction_baseline
from ml.classifier import train_logistic_regression, train_random_forest
from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.dataset import TARGET_COLUMN


MODEL_NAMES = (
    "majority_baseline",
    "market_direction_baseline",
    "logistic_regression",
    "random_forest",
)
CLASSIFIER_NAMES = ("logistic_regression", "random_forest")
BASELINE_NAMES = ("majority_baseline", "market_direction_baseline")


def compare_models(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: ModelConfig = DEFAULT_CONFIG,
) -> dict:
    """Run every method and compare their results on one test set."""

    results = {
        "majority_baseline": run_majority_baseline(training_data, test_data),
        "market_direction_baseline": run_market_direction_baseline(test_data),
        "logistic_regression": train_logistic_regression(
            training_data, test_data, config
        ),
        "random_forest": train_random_forest(training_data, test_data, config),
    }

    scores = pd.DataFrame(
        {name: result["metrics"] for name, result in results.items()}
    ).T
    scores.index.name = "model"

    targets = test_data[TARGET_COLUMN].astype("int64")
    matrices = {
        name: confusion_matrix(
            targets,
            result["predictions"],
            labels=[0, 1],
        )
        for name, result in results.items()
    }

    selected_model = max(
        CLASSIFIER_NAMES,
        key=lambda name: (
            scores.loc[name, "accuracy"],
            scores.loc[name, "f1"],
            -CLASSIFIER_NAMES.index(name),
        ),
    )
    best_baseline = max(
        BASELINE_NAMES,
        key=lambda name: (
            scores.loc[name, "accuracy"],
            scores.loc[name, "f1"],
            -BASELINE_NAMES.index(name),
        ),
    )

    return {
        "results": results,
        "scores": scores,
        "confusion_matrices": matrices,
        "selected_model": selected_model,
        "best_baseline": best_baseline,
        "beats_baseline": bool(
            scores.loc[selected_model, "accuracy"]
            > scores.loc[best_baseline, "accuracy"]
        ),
    }
