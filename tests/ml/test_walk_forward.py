import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.dataset import FEATURE_COLUMNS
from ml.walk_forward import WalkForwardError, run_walk_forward_evaluation


def model_data(rows=30):
    days = range(rows)
    data = {
        column: [day * (position + 1) / 100 for day in days]
        for position, column in enumerate(FEATURE_COLUMNS)
    }
    data["target_up"] = [day % 2 for day in days]
    return pd.DataFrame(
        data,
        index=pd.bdate_range("2025-01-01", periods=rows, name="Date"),
    )


@pytest.fixture(scope="module")
def walk_forward_result():
    return run_walk_forward_evaluation(
        model_data(), initial_train_ratio=0.6, block_size=4
    )


def test_every_row_after_initial_training_is_predicted_once(walk_forward_result):
    expected_index = model_data().index[18:]

    assert walk_forward_result["predictions"].index.equals(expected_index)
    assert walk_forward_result["probabilities"].index.equals(expected_index)
    assert len(walk_forward_result["predictions"]) == 12


def test_each_fold_trains_only_on_older_rows(walk_forward_result):
    for fold in walk_forward_result["folds"]:
        assert fold["training_end"] < fold["test_start"]


def test_training_data_grows_after_each_block(walk_forward_result):
    folds = walk_forward_result["folds"]

    assert [fold["training_rows"] for fold in folds] == [18, 22, 26]
    assert [fold["test_rows"] for fold in folds] == [4, 4, 4]


def test_last_block_can_be_smaller():
    result = run_walk_forward_evaluation(
        model_data(), initial_train_ratio=0.6, block_size=5
    )

    assert [fold["test_rows"] for fold in result["folds"]] == [5, 5, 2]
    assert len(result["predictions"]) == 12


def test_walk_forward_metrics_match_all_predictions(walk_forward_result):
    targets = model_data().loc[walk_forward_result["predictions"].index, "target_up"]
    predictions = walk_forward_result["predictions"]

    assert walk_forward_result["metrics"] == pytest.approx(
        {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, zero_division=0),
            "recall": recall_score(targets, predictions, zero_division=0),
            "f1": f1_score(targets, predictions, zero_division=0),
        }
    )


def test_walk_forward_does_not_change_the_dataset():
    dataset = model_data()
    before = dataset.copy(deep=True)

    run_walk_forward_evaluation(dataset, initial_train_ratio=0.6, block_size=5)

    pd.testing.assert_frame_equal(dataset, before)


@pytest.mark.parametrize("bad_size", [0, -1, 2.5, "5", True, None])
def test_bad_block_size_is_rejected(bad_size):
    with pytest.raises(WalkForwardError, match="positive whole number"):
        run_walk_forward_evaluation(model_data(), block_size=bad_size)
