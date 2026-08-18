"""Train and score the first direction classifier."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ml.config import DEFAULT_CONFIG, ModelConfig
from ml.dataset import FEATURE_COLUMNS, TARGET_COLUMN


class ClassifierError(ValueError):
    """Raised when a classifier cannot use the given data."""


def _read_features(data: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ClassifierError(f"{label} data must not be empty.")

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise ClassifierError(f"{label} data is missing features: {names}.")

    try:
        features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ClassifierError(f"{label} features must contain numbers.") from error

    if features.isna().any().any():
        raise ClassifierError(
            f"{label} features must not contain missing values."
        )

    if features.isin([float("inf"), float("-inf")]).any().any():
        raise ClassifierError(f"{label} features must contain finite numbers.")

    return features


def _read_targets(data: pd.DataFrame, label: str) -> pd.Series:
    if TARGET_COLUMN not in data.columns:
        raise ClassifierError(f"{label} data must contain {TARGET_COLUMN}.")

    targets = data[TARGET_COLUMN]
    if targets.isna().any():
        raise ClassifierError(f"{label} targets must not contain missing values.")

    if not set(targets.unique()).issubset({0, 1}):
        raise ClassifierError(f"{label} targets must contain only 0 and 1.")

    return targets.astype("int64")


def train_logistic_regression(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: ModelConfig = DEFAULT_CONFIG,
) -> dict:
    """Fit Logistic Regression on training data and score test predictions."""

    training_features = _read_features(training_data, "Training")
    test_features = _read_features(test_data, "Test")
    training_targets = _read_targets(training_data, "Training")
    test_targets = _read_targets(test_data, "Test")

    if training_targets.nunique() < 2:
        raise ClassifierError("Training targets must contain both 0 and 1.")

    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(training_features)
    scaled_test = scaler.transform(test_features)

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.random_seed,
    )
    model.fit(scaled_training, training_targets)

    predictions = pd.Series(
        model.predict(scaled_test),
        index=test_data.index.copy(),
        dtype="int64",
        name="logistic_prediction",
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
        "model": model,
        "scaler": scaler,
        "predictions": predictions,
        "metrics": metrics,
    }
