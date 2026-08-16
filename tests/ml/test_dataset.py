import pandas as pd
import pytest

from ml.dataset import (
    FEATURE_COLUMNS,
    MODEL_COLUMNS,
    DatasetError,
    build_model_dataset,
    get_dataset_summary,
)


def sample_prices(rows=30):
    closes = [100 + day + (day % 3) for day in range(rows)]
    return pd.DataFrame(
        {
            "Open": [close * 0.99 for close in closes],
            "High": [close * 1.02 for close in closes],
            "Low": [close * 0.98 for close in closes],
            "Close": closes,
            "Volume": [1000 + day * 25 for day in range(rows)],
        },
        index=pd.bdate_range("2025-01-01", periods=rows),
    )


def test_dataset_contains_model_columns_only():
    dataset = build_model_dataset(sample_prices())

    assert dataset.columns.tolist() == MODEL_COLUMNS
    assert len(FEATURE_COLUMNS) == 14


def test_dataset_removes_unfinished_rows():
    dataset = build_model_dataset(sample_prices(30))

    assert len(dataset) == 9
    assert dataset.index[0] == sample_prices(30).index[20]
    assert dataset.index[-1] == sample_prices(30).index[-2]


def test_dataset_has_no_missing_values():
    dataset = build_model_dataset(sample_prices())

    assert not dataset.isna().any().any()


def test_dataset_dates_are_sorted():
    prices = sample_prices().sort_index(ascending=False)

    dataset = build_model_dataset(prices)

    assert dataset.index.is_monotonic_increasing


def test_dataset_target_contains_only_zero_and_one():
    dataset = build_model_dataset(sample_prices())

    assert set(dataset["target_up"].unique()).issubset({0, 1})
    assert dataset["target_up"].dtype == "int64"


def test_dataset_does_not_change_original_prices():
    original = sample_prices()
    original_before = original.copy(deep=True)

    build_model_dataset(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_small_dataset_has_clear_error():
    with pytest.raises(DatasetError, match="Not enough price rows"):
        build_model_dataset(sample_prices(20))


def test_dataset_summary_is_correct():
    dataset = build_model_dataset(sample_prices())

    summary = get_dataset_summary(dataset)

    assert summary["rows"] == len(dataset)
    assert summary["start_date"] == dataset.index.min().date().isoformat()
    assert summary["end_date"] == dataset.index.max().date().isoformat()
    assert summary["feature_count"] == 14
    assert summary["up_days"] + summary["not_up_days"] == len(dataset)


def test_summary_rejects_empty_dataset():
    with pytest.raises(DatasetError, match="must not be empty"):
        get_dataset_summary(pd.DataFrame())


def test_summary_requires_target():
    dataset = pd.DataFrame(
        {"return_1d": [0.01]},
        index=pd.to_datetime(["2025-01-01"]),
    )

    with pytest.raises(DatasetError, match="must contain target_up"):
        get_dataset_summary(dataset)
