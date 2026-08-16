import pandas as pd
import pytest

from ml.split import SplitError, get_split_summary, split_dataset


def sample_dataset(rows=10):
    return pd.DataFrame(
        {
            "return_1d": [day / 100 for day in range(rows)],
            "target_up": [day % 2 for day in range(rows)],
        },
        index=pd.bdate_range("2025-01-01", periods=rows),
    )


def test_default_split_is_eighty_twenty():
    training_data, test_data = split_dataset(sample_dataset())

    assert len(training_data) == 8
    assert len(test_data) == 2


def test_training_dates_come_before_test_dates():
    training_data, test_data = split_dataset(sample_dataset())

    assert training_data.index.max() < test_data.index.min()


def test_no_rows_appear_in_both_parts():
    training_data, test_data = split_dataset(sample_dataset())

    assert training_data.index.intersection(test_data.index).empty


def test_no_rows_are_lost():
    original = sample_dataset()

    training_data, test_data = split_dataset(original)
    joined = pd.concat([training_data, test_data])

    pd.testing.assert_frame_equal(joined, original)


def test_split_does_not_change_original_dataset():
    original = sample_dataset()
    original_before = original.copy(deep=True)

    split_dataset(original)

    pd.testing.assert_frame_equal(original, original_before)


def test_custom_training_ratio_is_used():
    training_data, test_data = split_dataset(
        sample_dataset(20), train_ratio=0.75
    )

    assert len(training_data) == 15
    assert len(test_data) == 5


def test_shuffled_dataset_is_rejected():
    shuffled = sample_dataset().sample(frac=1, random_state=42)

    with pytest.raises(SplitError, match="sorted from oldest to newest"):
        split_dataset(shuffled)


def test_duplicate_dates_are_rejected():
    dataset = sample_dataset()
    dataset.index = dataset.index[:-1].append(dataset.index[-2:-1])

    with pytest.raises(SplitError, match="duplicate dates"):
        split_dataset(dataset)


def test_small_dataset_is_rejected():
    with pytest.raises(SplitError, match="at least 10 rows"):
        split_dataset(sample_dataset(9))


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1, "0.8", True, None])
def test_bad_training_ratio_is_rejected(ratio):
    with pytest.raises(SplitError, match="number between 0 and 1"):
        split_dataset(sample_dataset(), train_ratio=ratio)


def test_non_date_index_is_rejected():
    dataset = sample_dataset().reset_index(drop=True)

    with pytest.raises(SplitError, match="must use a date index"):
        split_dataset(dataset)


def test_split_summary_has_dates_and_counts():
    training_data, test_data = split_dataset(sample_dataset())

    summary = get_split_summary(training_data, test_data)

    assert summary == {
        "training_rows": 8,
        "training_start": "2025-01-01",
        "training_end": "2025-01-10",
        "test_rows": 2,
        "test_start": "2025-01-13",
        "test_end": "2025-01-14",
    }


def test_summary_rejects_an_empty_part():
    with pytest.raises(SplitError, match="both contain rows"):
        get_split_summary(sample_dataset(), pd.DataFrame())
