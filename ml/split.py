"""Split model data into older training rows and newer test rows."""

import pandas as pd


DEFAULT_TRAIN_RATIO = 0.8
MINIMUM_DATASET_ROWS = 10


class SplitError(ValueError):
    """Raised when a safe time-based split cannot be made."""


def split_dataset(
    dataset: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return older training rows and newer test rows without shuffling."""

    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise SplitError("Model dataset must not be empty.")

    if len(dataset) < MINIMUM_DATASET_ROWS:
        raise SplitError(
            f"Model dataset must contain at least {MINIMUM_DATASET_ROWS} rows."
        )

    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise SplitError("Model dataset must use a date index.")

    if dataset.index.hasnans:
        raise SplitError("Model dataset index must not contain missing dates.")

    if dataset.index.duplicated().any():
        raise SplitError("Model dataset must not contain duplicate dates.")

    if not dataset.index.is_monotonic_increasing:
        raise SplitError("Model dataset must be sorted from oldest to newest.")

    if (
        isinstance(train_ratio, bool)
        or not isinstance(train_ratio, (int, float))
        or not 0 < train_ratio < 1
    ):
        raise SplitError("Training ratio must be a number between 0 and 1.")

    split_position = int(len(dataset) * train_ratio)
    if split_position == 0 or split_position == len(dataset):
        raise SplitError("Training and test data must both contain rows.")

    training_data = dataset.iloc[:split_position].copy()
    test_data = dataset.iloc[split_position:].copy()
    return training_data, test_data


def get_split_summary(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    """Return the row counts and date ranges for a completed split."""

    if training_data.empty or test_data.empty:
        raise SplitError("Training and test data must both contain rows.")

    return {
        "training_rows": len(training_data),
        "training_start": training_data.index.min().date().isoformat(),
        "training_end": training_data.index.max().date().isoformat(),
        "test_rows": len(test_data),
        "test_start": test_data.index.min().date().isoformat(),
        "test_end": test_data.index.max().date().isoformat(),
    }
