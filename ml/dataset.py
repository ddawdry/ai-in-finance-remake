"""Build the clean table used for model training."""

import pandas as pd

from ml.features import (
    add_direction_target,
    add_lagged_returns,
    add_moving_averages,
    add_rolling_volatility,
    add_volume_and_range_features,
)
from ml.validation import validate_prices


FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "ma_5d",
    "ma_10d",
    "ma_20d",
    "close_to_ma_5d",
    "close_to_ma_10d",
    "close_to_ma_20d",
    "volatility_5d",
    "volatility_20d",
    "volume_change_1d",
    "daily_range",
    "open_to_close",
]
TARGET_COLUMN = "target_up"
MODEL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


class DatasetError(ValueError):
    """Raised when a model dataset cannot be built."""


def build_model_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate prices and build a finished modelling dataset."""

    data = validate_prices(prices)
    data = add_lagged_returns(data)
    data = add_moving_averages(data)
    data = add_rolling_volatility(data)
    data = add_volume_and_range_features(data)
    data = add_direction_target(data)

    dataset = data[MODEL_COLUMNS].dropna().copy()

    if dataset.empty:
        raise DatasetError(
            "Not enough price rows were provided to build the model dataset."
        )

    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype("int64")
    return dataset


def get_dataset_summary(dataset: pd.DataFrame) -> dict:
    """Return a short summary of a finished model dataset."""

    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise DatasetError("Model dataset must not be empty.")

    if TARGET_COLUMN not in dataset.columns:
        raise DatasetError("Model dataset must contain target_up.")

    return {
        "rows": len(dataset),
        "start_date": dataset.index.min().date().isoformat(),
        "end_date": dataset.index.max().date().isoformat(),
        "feature_count": len(FEATURE_COLUMNS),
        "up_days": int((dataset[TARGET_COLUMN] == 1).sum()),
        "not_up_days": int((dataset[TARGET_COLUMN] == 0).sum()),
    }
