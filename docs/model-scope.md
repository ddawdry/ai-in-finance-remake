# Direction Model Scope

## Aim

The model predicts whether an asset will close up or down on the next day.

It does not predict an exact closing price. Its main outputs are an up or down label and an estimated chance of an up day.

## Market Data

Daily market data comes from Yahoo Finance through `yfinance`. The default date range starts on `2016-01-01` and ends at the latest available day.

The project currently supports these assets:

**Stocks:** AAPL, MSFT, TSLA, GOOGL, and AMZN

**Crypto:** BTC-USD, ETH-USD, SOL-USD, ADA-USD, and DOGE-USD

Stock data follows exchange trading days. Crypto data can include weekends because those markets run every day.

## Prediction Target

For each row, the model compares the current closing price with the next closing price.

- `1` means the next close is higher.
- `0` means the next close is the same or lower.

The final day is removed because its next closing price is not known. Model features only use the current day and earlier days. They do not use information from the day being predicted.

## Model Features

The model uses 14 features made from historical price and volume data:

- Returns over 1, 3, and 5 days
- Moving averages over 5, 10, and 20 days
- Closing price compared with each moving average
- Volatility over 5 and 20 days
- One-day volume change
- Daily high-to-low price range
- Open-to-close price change

Rows with missing feature values are removed before training. This mainly affects the first rows because rolling features need earlier data.

## Classifier

The main model is a Random Forest classifier from scikit-learn.

Its current settings are:

- 200 decision trees
- Maximum tree depth of 6
- Random seed of 42
- One processing job, which keeps test runs repeatable

The model returns a direction label and an up probability for each test row.

## Model Evaluation

The data always stays in date order and is never randomly shuffled.

The oldest 80% of rows form the first training set. The model then predicts the next block of up to 60 rows. After each block, those dates become part of the training history before the following block is tested. This is called walk-forward evaluation.

The reported model measurements are:

- Accuracy
- Precision
- Recall
- F1 score

The exported result is compared with a majority-class baseline. This baseline predicts the most common direction found in the original training data for every test row.

## Backtest

The backtest uses the walk-forward predictions. The strategy holds the asset for the next day when the model predicts up. It stays out of the market when the model predicts down.

The strategy is compared with buying and holding the same asset over the matching dates. A trading cost of `0.1%` is applied whenever the strategy changes position.

The backtest reports:

- Total return
- Annualised return
- Annualised volatility
- Maximum drawdown
- Sharpe ratio

Stocks use 252 trading days per year for annual calculations. Crypto uses 365 days.

This is a simplified historical test. It does not include every real trading cost, delay, spread, or tax.

## Outside the Current Scope

The current version does not include:

- Exact price predictions
- Live or intraday trading
- Automatic buying or selling
- Managing investments for users
- Portfolio building or position sizing
- Options, futures, or leveraged products
- Paid market data or paid APIs
- News or social media sentiment
- Deep learning
- Claims that the model will make a profit
- Financial advice

## What Counts as a Useful Result

The model does not need unusually high accuracy to make this project useful. The main goal is to build a repeatable process, prevent future data from leaking into training, compare the model fairly, and explain the results honestly.

The full pipeline is explained in more detail in [model.md](model.md).
