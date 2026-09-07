# Model Guide

This document explains how the direction model works from the first data download to the results shown in the dashboard.

## Model Goal

The model answers one question:

> Is the asset more likely to close up or down on the next day?

It is a binary classification problem. The model does not predict the next closing price or decide whether somebody should make a trade.

## Pipeline Overview

The main pipeline runs in this order:

```text
Download prices
      |
Validate and cache the data
      |
Create model features and the target
      |
Run walk-forward evaluation
      |
Compare with the baseline
      |
Run the historical backtest
      |
Save results for the API and dashboard
```

The full pipeline for one asset is in `ml/model.py`. The multi-asset runner is in `ml/run_assets.py`.

## Price Data

Historical data is downloaded from Yahoo Finance with `yfinance`. The model uses adjusted daily values for:

- Open
- High
- Low
- Close
- Volume

The default date range starts on `2016-01-01` and ends at the latest available day. The default ticker is AAPL.

Downloaded data is stored in `data/raw/`. If a matching cache file already exists, the project uses it instead of downloading the same data again. Passing `--refresh` asks for a fresh download.

The downloaded files are ignored by Git. This means each person running the project creates their own local market data.

## Data Checks

The project checks the price data before creating any features. It rejects data when:

- A required column is missing
- A date cannot be read
- A date is missing or repeated
- A required value is missing or is not a number
- A price is zero or negative
- Volume is negative
- The daily high and low values do not make sense

Valid rows are sorted from oldest to newest. The original DataFrame is not changed during validation.

These checks help catch bad input early. They do not prove that every value supplied by the data source is correct.

## Model Features

The finished dataset contains 14 features. Every feature is made from the current day or earlier days.

| Feature | Meaning |
| --- | --- |
| `return_1d` | Closing price return over 1 day |
| `return_3d` | Closing price return over 3 days |
| `return_5d` | Closing price return over 5 days |
| `ma_5d` | Average closing price over 5 days |
| `ma_10d` | Average closing price over 10 days |
| `ma_20d` | Average closing price over 20 days |
| `close_to_ma_5d` | Closing price divided by the 5-day average |
| `close_to_ma_10d` | Closing price divided by the 10-day average |
| `close_to_ma_20d` | Closing price divided by the 20-day average |
| `volatility_5d` | Standard deviation of daily returns over 5 days |
| `volatility_20d` | Standard deviation of daily returns over 20 days |
| `volume_change_1d` | Change from the previous day's volume |
| `daily_range` | Difference between the high and low, divided by the low |
| `open_to_close` | Difference between the close and open, divided by the open |

The first rows do not have enough history for the rolling features. Those rows are removed after all features have been created.

## Prediction Target

The target column is called `target_up`.

- `1` means the next closing price is higher than the current closing price.
- `0` means the next closing price is equal to or lower than the current closing price.

The final price row has no known next day, so it cannot have a target and is removed.

The target is kept separate from the feature columns. This prevents the model from using the answer as an input.

## Random Forest Classifier

The main model is scikit-learn's `RandomForestClassifier`. A Random Forest combines the output of several decision trees.

The current settings are:

| Setting | Value |
| --- | --- |
| Number of trees | 200 |
| Maximum tree depth | 6 |
| Random seed | 42 |
| Processing jobs | 1 |

The fixed random seed makes repeated tests easier to compare. Limiting the tree depth also reduces how much detail each tree can memorise from the training data.

For each test row, the model returns an up or down prediction and an estimated up probability. This probability is the Random Forest's estimate. It is not a promise, and the project does not currently test whether those probability values are perfectly calibrated.

## Date-Based Splitting

Financial data must be tested in time order. Randomly mixing old and new dates could let the model learn from information that belongs in the future.

The project uses the oldest 80% of the prepared rows as the first training set. The newer 20% is kept for testing.

No rows are shuffled. The training dates must finish before the test dates begin.

## Walk-Forward Evaluation

The main evaluation goes through the test period in blocks of up to 60 rows.

For each block:

1. The model trains on all earlier rows.
2. It predicts the next block of dates.
3. The block is added to the available history.
4. A new model is trained before predicting the following block.

This gives every prediction a realistic order: training always happens with past data and testing always happens with later data.

The process is still simpler than a live system. It does not model training time, data delays, or changes made after seeing previous results.

## Baseline

A model score needs a simple comparison. The main pipeline uses a majority-class baseline.

The baseline finds the most common target in the original training data and predicts that same direction for every test row. For example, if up days are more common in training, it always predicts up.

Beating this baseline does not prove that the model will work in the future. It only shows whether the model did better than a very simple rule on the same test period.

## Classification Measurements

The model and baseline are compared with four measurements:

- **Accuracy:** the share of all directions predicted correctly
- **Precision:** how often a predicted up day was actually up
- **Recall:** how many of the real up days were found
- **F1 score:** a balance between precision and recall

Looking at more than one measurement gives a better view than accuracy alone, especially when one direction happens more often than the other.

## Historical Backtest

The backtest uses only the predictions made during walk-forward testing.

The simple strategy works like this:

- When the model predicts up, the strategy receives the next day's asset return.
- When the model predicts down, the strategy stays out of the market for that day.

It is compared with buying and holding the same asset over the same dates.

A cost of `0.001`, or 0.1%, is removed whenever the strategy changes between being in and out of the market. This is a basic cost estimate rather than a full trading cost model.

The backtest calculates:

- Total return
- Annualised return
- Annualised volatility
- Maximum drawdown
- Sharpe ratio

Annual values use 252 days for stocks and 365 days for crypto. The risk-free rate is currently set to zero.

The backtest does not include spreads, slippage, tax, rejected orders, liquidity problems, or every other issue found in real trading.

## Saved Results

The model creates these files:

| File | Contents |
| --- | --- |
| `direction_predictions.csv` | Date, actual direction, predicted direction, and up probability |
| `model_metrics.json` | Model and baseline measurements |
| `model_results.json` | Metrics and prediction rows used by the API |
| `backtest_results.json` | Backtest settings, summary measurements, and chart data |

Running `python -m ml.model` saves the AAPL results directly under `data/results/`.

Running `python -m ml.run_assets` gives each asset its own folder, such as `data/results/AAPL/` or `data/results/BTC-USD/`.

Generated results are ignored by Git because they can be recreated from the model pipeline.

## Repeatability

The code uses a fixed model seed and keeps dates in a fixed order. The test suite also checks data validation, feature calculations, leakage rules, classifiers, baselines, walk-forward evaluation, backtesting, and result export. The commands and test groups are listed in [testing.md](testing.md).

Results can still change when Yahoo Finance updates its historical data or when package versions change. A result should always be saved with enough information to explain when and how it was created.

## Limits

This model only learns from the price and volume features listed above. It does not use company reports, news, economic announcements, market sentiment, or knowledge of unusual events.

Historical patterns may stop working. A good test result can also happen by chance. The model should be treated as a learning exercise and not as evidence that future returns can be predicted reliably.

This project does not provide financial advice and should not be used to make real investment decisions.
