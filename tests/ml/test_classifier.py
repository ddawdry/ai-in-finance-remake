import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.classifier import ClassifierError, train_logistic_regression
from ml.dataset import FEATURE_COLUMNS


def model_data(rows=30, start="2025-01-01", feature_offset=0):
    values = range(rows)
    data = {
        column: [
            feature_offset + day * (position + 1) / 100
            for day in values
        ]
        for position, column in enumerate(FEATURE_COLUMNS)
    }
    data["target_up"] = [day % 2 for day in values]
    return pd.DataFrame(
        data,
        index=pd.bdate_range(start, periods=rows),
    )


def test_creates_one_prediction_for_each_test_row():
    test_data = model_data(8, start="2025-04-01")

    result = train_logistic_regression(model_data(), test_data)

    assert len(result["predictions"]) == len(test_data)
    assert result["predictions"].index.equals(test_data.index)
    assert result["predictions"].name == "logistic_prediction"


def test_predictions_only_contain_zero_and_one():
    result = train_logistic_regression(
        model_data(), model_data(8, start="2025-04-01")
    )

    assert set(result["predictions"].unique()).issubset({0, 1})


def test_scaler_is_fitted_on_training_features_only():
    training_data = model_data()
    test_data = model_data(8, start="2025-04-01", feature_offset=1000)

    result = train_logistic_regression(training_data, test_data)

    expected_means = training_data[FEATURE_COLUMNS].mean().to_numpy()
    combined_means = pd.concat(
        [training_data, test_data]
    )[FEATURE_COLUMNS].mean().to_numpy()
    assert result["scaler"].mean_ == pytest.approx(expected_means)
    assert result["scaler"].mean_.tolist() != pytest.approx(combined_means)


def test_metrics_match_the_predictions():
    test_data = model_data(8, start="2025-04-01")

    result = train_logistic_regression(model_data(), test_data)
    predictions = result["predictions"]
    targets = test_data["target_up"]

    assert result["metrics"] == pytest.approx(
        {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, zero_division=0),
            "recall": recall_score(targets, predictions, zero_division=0),
            "f1": f1_score(targets, predictions, zero_division=0),
        }
    )


def test_same_data_gives_repeatable_results():
    training_data = model_data()
    test_data = model_data(8, start="2025-04-01")

    first = train_logistic_regression(training_data, test_data)
    second = train_logistic_regression(training_data, test_data)

    pd.testing.assert_series_equal(first["predictions"], second["predictions"])
    assert first["metrics"] == second["metrics"]


def test_classifier_does_not_change_original_data():
    training_data = model_data()
    test_data = model_data(8, start="2025-04-01")
    training_before = training_data.copy(deep=True)
    test_before = test_data.copy(deep=True)

    train_logistic_regression(training_data, test_data)

    pd.testing.assert_frame_equal(training_data, training_before)
    pd.testing.assert_frame_equal(test_data, test_before)


def test_missing_feature_is_rejected():
    training_data = model_data().drop(columns=FEATURE_COLUMNS[0])

    with pytest.raises(ClassifierError, match="missing features"):
        train_logistic_regression(
            training_data,
            model_data(8, start="2025-04-01"),
        )


@pytest.mark.parametrize("bad_value", [None, "unknown", float("inf")])
def test_bad_feature_value_is_rejected(bad_value):
    training_data = model_data()
    if isinstance(bad_value, str):
        training_data[FEATURE_COLUMNS[0]] = training_data[
            FEATURE_COLUMNS[0]
        ].astype(object)
    training_data.loc[training_data.index[0], FEATURE_COLUMNS[0]] = bad_value
    message = {
        None: "missing values",
        "unknown": "numbers",
        float("inf"): "finite numbers",
    }[bad_value]

    with pytest.raises(ClassifierError, match=message):
        train_logistic_regression(
            training_data,
            model_data(8, start="2025-04-01"),
        )


def test_missing_target_is_rejected():
    test_data = model_data(8, start="2025-04-01").drop(columns="target_up")

    with pytest.raises(ClassifierError, match="must contain target_up"):
        train_logistic_regression(model_data(), test_data)


def test_invalid_target_is_rejected():
    test_data = model_data(8, start="2025-04-01")
    test_data.loc[test_data.index[0], "target_up"] = 2

    with pytest.raises(ClassifierError, match="only 0 and 1"):
        train_logistic_regression(model_data(), test_data)


def test_training_data_needs_both_target_classes():
    training_data = model_data()
    training_data["target_up"] = 1

    with pytest.raises(ClassifierError, match="must contain both 0 and 1"):
        train_logistic_regression(
            training_data,
            model_data(8, start="2025-04-01"),
        )
