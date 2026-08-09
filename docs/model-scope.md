# Direction Model Scope

## Aim

The first model will predict whether a stock closes up or down on the next trading day.

It will not try to predict the exact closing price. The main result will be a direction label and, later, the model's estimated chance of an up day.

## First Dataset

The first version will use:

- Ticker: `AAPL`
- Start date: `2016-01-01`
- End date: the latest completed trading day
- Interval: one day
- Data source: Yahoo Finance through `yfinance`

Starting with one stock will make it easier to check the data, features, model, and backtest. More stocks and cryptocurrencies can be added after the first version works properly.

## Prediction Target

For each trading day, the model will compare that day's closing price with the next trading day's closing price.

- `1` means the next close is higher.
- `0` means the next close is the same or lower.

The model will only use information that was available by the end of the current day. It must not use prices or other values from the day it is trying to predict.

## First Features

The first feature set will be based on historical price and volume data. It will include:

- Lagged returns
- Moving averages
- Price compared with moving averages
- Rolling volatility
- Daily price range
- Volume changes where the data is available

Each feature will be calculated from the current day and earlier days only.

## How It Will Be Checked

The data will stay in date order. Older rows will be used for training and newer rows will be used for testing.

The model will be compared with simple baselines. One baseline will predict the most common direction in the training data. Another will predict that the next day follows the current day's direction.

A backtest will compare the model's signals with buying and holding the same stock. Trading costs will be included later so the result is not too generous.

## Not Included in the First Version

The first version will not include:

- Exact price predictions
- Intraday or live trading
- Automatic buying or selling
- Options, futures, or leveraged products
- Portfolio building or position sizing
- Paid market data or paid APIs
- News or social media sentiment
- Deep learning
- Claims that the model can make a profit
- Financial advice

## What Counts as a Useful Result

The model does not need high accuracy to make this project useful. The first version will be successful if the data steps are repeatable, the model does not use future information, the backtest is fair, and the results are compared honestly with the baselines.
