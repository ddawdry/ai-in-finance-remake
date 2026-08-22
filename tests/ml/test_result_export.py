import json

import pandas as pd
import pytest

from ml.result_export import ResultExportError, export_direction_results


def model_dataset():
    return pd.DataFrame(
        {"target_up": [0, 1, 1, 0]},
        index=pd.date_range("2025-01-01", periods=4, name="Date"),
    )


def model_evaluation():
    dates = model_dataset().index[-3:]
    return {
        "predictions": pd.Series([1, 1, 0], index=dates, dtype="int64"),
        "probabilities": pd.Series([0.7, 0.8, 0.4], index=dates),
        "metrics": {
            "accuracy": 2 / 3,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2 / 3,
        },
    }


def test_export_creates_csv_and_json_files(tmp_path):
    paths = export_direction_results(
        model_dataset(), model_evaluation(), "AAPL", "stock", tmp_path
    )

    assert paths["predictions_path"].is_file()
    assert paths["metrics_path"].is_file()


def test_csv_contains_direction_results_only(tmp_path):
    paths = export_direction_results(
        model_dataset(), model_evaluation(), "AAPL", "stock", tmp_path
    )
    rows = pd.read_csv(paths["predictions_path"])

    assert rows.columns.tolist() == [
        "date",
        "ticker",
        "asset_type",
        "actual_direction",
        "predicted_direction",
        "up_probability",
    ]
    assert rows["actual_direction"].tolist() == ["up", "up", "down"]
    assert rows["predicted_direction"].tolist() == ["up", "up", "down"]
    assert rows["up_probability"].tolist() == pytest.approx([0.7, 0.8, 0.4])
    assert not any("price" in column.lower() for column in rows.columns)


def test_json_contains_metrics_and_basic_details(tmp_path):
    paths = export_direction_results(
        model_dataset(), model_evaluation(), "btc-usd", "crypto", tmp_path
    )
    with paths["metrics_path"].open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["ticker"] == "BTC-USD"
    assert result["asset_type"] == "crypto"
    assert result["model"] == "random_forest_walk_forward"
    assert result["prediction_rows"] == 3
    assert result["start_date"] == "2025-01-02"
    assert result["end_date"] == "2025-01-04"
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_export_does_not_change_input_data(tmp_path):
    dataset = model_dataset()
    evaluation = model_evaluation()
    dataset_before = dataset.copy(deep=True)
    predictions_before = evaluation["predictions"].copy(deep=True)

    export_direction_results(dataset, evaluation, "AAPL", "stock", tmp_path)

    pd.testing.assert_frame_equal(dataset, dataset_before)
    pd.testing.assert_series_equal(
        evaluation["predictions"], predictions_before
    )


def test_prediction_and_probability_dates_must_match(tmp_path):
    evaluation = model_evaluation()
    evaluation["probabilities"].index = pd.date_range(
        "2026-01-01", periods=3
    )

    with pytest.raises(ResultExportError, match="dates must match"):
        export_direction_results(
            model_dataset(), evaluation, "AAPL", "stock", tmp_path
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, None])
def test_bad_probabilities_are_rejected(tmp_path, probability):
    evaluation = model_evaluation()
    evaluation["probabilities"].iloc[0] = probability

    with pytest.raises(ResultExportError, match="between 0 and 1"):
        export_direction_results(
            model_dataset(), evaluation, "AAPL", "stock", tmp_path
        )


@pytest.mark.parametrize("asset_type", ["fund", "", None])
def test_bad_asset_type_is_rejected(tmp_path, asset_type):
    with pytest.raises(ResultExportError, match="Asset type"):
        export_direction_results(
            model_dataset(), model_evaluation(), "AAPL", asset_type, tmp_path
        )


def test_unsafe_ticker_text_is_rejected(tmp_path):
    with pytest.raises(ResultExportError, match="Ticker"):
        export_direction_results(
            model_dataset(), model_evaluation(), "=FORMULA", "stock", tmp_path
        )
