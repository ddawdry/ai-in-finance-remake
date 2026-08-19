import numpy as np
import pandas as pd

from ml.dataset import FEATURE_COLUMNS
from ml.evaluation import CLASSIFIER_NAMES, MODEL_NAMES, compare_models


def model_data(rows=40, start="2025-01-01", offset=0):
    days = np.arange(rows)
    data = {
        column: offset + np.sin(days + position) + days / 100
        for position, column in enumerate(FEATURE_COLUMNS)
    }
    data["return_1d"] = np.where(days % 3 == 0, 0.01, -0.01)
    data["target_up"] = (days % 2).astype("int64")
    return pd.DataFrame(data, index=pd.bdate_range(start, periods=rows))


def test_all_models_are_compared():
    result = compare_models(
        model_data(), model_data(12, "2025-04-01", offset=2)
    )

    assert tuple(result["scores"].index) == MODEL_NAMES
    assert set(result["results"]) == set(MODEL_NAMES)


def test_score_table_contains_the_expected_metrics():
    result = compare_models(
        model_data(), model_data(12, "2025-04-01", offset=2)
    )

    assert result["scores"].columns.tolist() == [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    assert result["scores"].map(lambda value: 0 <= value <= 1).all().all()


def test_each_model_has_a_two_by_two_confusion_matrix():
    result = compare_models(
        model_data(), model_data(12, "2025-04-01", offset=2)
    )

    assert set(result["confusion_matrices"]) == set(MODEL_NAMES)
    for matrix in result["confusion_matrices"].values():
        assert matrix.shape == (2, 2)
        assert matrix.sum() == 12


def test_confusion_matrix_uses_down_then_up_order():
    test_data = model_data(12, "2025-04-01", offset=2)

    result = compare_models(model_data(), test_data)
    predictions = result["results"]["majority_baseline"]["predictions"]
    expected = np.array(
        [
            [((test_data["target_up"] == 0) & (predictions == 0)).sum(),
             ((test_data["target_up"] == 0) & (predictions == 1)).sum()],
            [((test_data["target_up"] == 1) & (predictions == 0)).sum(),
             ((test_data["target_up"] == 1) & (predictions == 1)).sum()],
        ]
    )

    np.testing.assert_array_equal(
        result["confusion_matrices"]["majority_baseline"], expected
    )


def test_selected_model_is_the_best_classifier():
    result = compare_models(
        model_data(), model_data(12, "2025-04-01", offset=2)
    )
    scores = result["scores"]
    selected = result["selected_model"]

    classifier_scores = scores.loc[list(CLASSIFIER_NAMES)]
    assert (
        scores.loc[selected, "accuracy"]
        == classifier_scores["accuracy"].max()
    )
    tied = classifier_scores[
        classifier_scores["accuracy"] == scores.loc[selected, "accuracy"]
    ]
    assert scores.loc[selected, "f1"] == tied["f1"].max()


def test_result_says_whether_the_classifier_beats_the_best_baseline():
    result = compare_models(
        model_data(), model_data(12, "2025-04-01", offset=2)
    )
    scores = result["scores"]

    expected = (
        scores.loc[result["selected_model"], "accuracy"]
        > scores.loc[result["best_baseline"], "accuracy"]
    )
    assert result["beats_baseline"] == expected


def test_comparison_does_not_change_the_input_data():
    training_data = model_data()
    test_data = model_data(12, "2025-04-01", offset=2)
    training_before = training_data.copy(deep=True)
    test_before = test_data.copy(deep=True)

    compare_models(training_data, test_data)

    pd.testing.assert_frame_equal(training_data, training_before)
    pd.testing.assert_frame_equal(test_data, test_before)
