import pandas as pd
import pytest

from ml.baseline import BaselineError, run_majority_baseline


def target_data(targets, start="2025-01-01"):
    return pd.DataFrame(
        {"target_up": targets},
        index=pd.bdate_range(start, periods=len(targets)),
    )


def test_most_common_training_target_is_chosen():
    result = run_majority_baseline(
        target_data([1, 1, 1, 0]),
        target_data([0, 1], start="2025-02-01"),
    )

    assert result["majority_class"] == 1


def test_test_targets_do_not_choose_the_majority_class():
    result = run_majority_baseline(
        target_data([1, 1, 1, 0]),
        target_data([0, 0, 0, 0], start="2025-02-01"),
    )

    assert result["majority_class"] == 1
    assert result["predictions"].eq(1).all()


def test_one_prediction_is_created_for_each_test_row():
    test_data = target_data([0, 1, 0], start="2025-02-01")

    result = run_majority_baseline(target_data([1, 1, 0]), test_data)

    assert len(result["predictions"]) == len(test_data)
    assert result["predictions"].index.equals(test_data.index)
    assert result["predictions"].name == "majority_prediction"


def test_metrics_are_correct():
    training_data = target_data([1, 1, 1, 0])
    test_data = target_data([1, 1, 0, 0], start="2025-02-01")

    result = run_majority_baseline(training_data, test_data)

    assert result["metrics"]["accuracy"] == pytest.approx(0.5)
    assert result["metrics"]["precision"] == pytest.approx(0.5)
    assert result["metrics"]["recall"] == pytest.approx(1.0)
    assert result["metrics"]["f1"] == pytest.approx(2 / 3)


def test_tied_training_targets_choose_zero():
    result = run_majority_baseline(
        target_data([0, 1, 0, 1]),
        target_data([0, 1], start="2025-02-01"),
    )

    assert result["majority_class"] == 0
    assert result["predictions"].eq(0).all()


def test_zero_predictions_have_safe_zero_metrics():
    result = run_majority_baseline(
        target_data([0, 0, 1]),
        target_data([1, 1], start="2025-02-01"),
    )

    assert result["metrics"] == {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


@pytest.mark.parametrize("bad_targets", [[0, None], [0, 2], ["0", "1"]])
def test_bad_training_targets_are_rejected(bad_targets):
    message = "missing values" if None in bad_targets else "only 0 and 1"

    with pytest.raises(BaselineError, match=message):
        run_majority_baseline(
            target_data(bad_targets),
            target_data([0, 1], start="2025-02-01"),
        )


def test_missing_target_column_is_rejected():
    training_data = pd.DataFrame(
        {"return_1d": [0.01, 0.02]},
        index=pd.bdate_range("2025-01-01", periods=2),
    )

    with pytest.raises(BaselineError, match="must contain target_up"):
        run_majority_baseline(
            training_data,
            target_data([0, 1], start="2025-02-01"),
        )


def test_empty_test_data_is_rejected():
    with pytest.raises(BaselineError, match="Test data must not be empty"):
        run_majority_baseline(target_data([0, 1]), pd.DataFrame())


def test_baseline_does_not_change_original_data():
    training_data = target_data([1, 1, 0])
    test_data = target_data([0, 1], start="2025-02-01")
    training_before = training_data.copy(deep=True)
    test_before = test_data.copy(deep=True)

    run_majority_baseline(training_data, test_data)

    pd.testing.assert_frame_equal(training_data, training_before)
    pd.testing.assert_frame_equal(test_data, test_before)
