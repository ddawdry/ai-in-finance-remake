# AI in Finance

AI in Finance is a learning project that uses historical market data to predict whether an asset may move up or down on the next day.

The project does not try to predict an exact future price. It looks for patterns in past data, tests those patterns on later data, and compares the model with a simple baseline.

## Project Background

This project started as a university group project between four students. The work was split into four main areas:

- AI and machine learning
- Frontend design
- Finding reliable market data
- Testing, data input, and data sanitisation

My main role was testing, data input, and data sanitisation. I am now rebuilding the project by myself so I can improve it and learn how every part works.

## Current Features

- Downloads free daily market data with `yfinance`
- Supports both stocks and cryptocurrencies
- Checks and cleans the downloaded data
- Creates lagged return, moving average, volatility, volume, and price range features
- Trains a Random Forest classifier
- Predicts next-day direction as up or down
- Uses walk-forward testing to keep the evaluation in date order
- Compares the model with a majority-class baseline
- Backtests a simple predicted-up strategy against buy and hold
- Includes a trading cost in the backtest
- Displays the results in a React dashboard

## Supported Assets

The current asset list includes:

**Stocks:** AAPL, MSFT, TSLA, GOOGL, and AMZN

**Crypto:** BTC-USD, ETH-USD, SOL-USD, ADA-USD, and DOGE-USD

An asset can only be selected in the dashboard after its model and backtest results have been created locally.

## How It Works

1. Historical daily prices are downloaded from Yahoo Finance.
2. The data is checked before it is used.
3. Features are created from past price and volume data.
4. The target records whether the next closing price went up or down.
5. A Random Forest classifier is trained and tested in date order.
6. The result is compared with a majority-class baseline.
7. A historical backtest compares the direction strategy with buy and hold.
8. The results are saved locally and shown by the dashboard.

## Setup

You will need:

- Python 3
- Node.js and npm
- An internet connection for the first market data download

Clone the repository and move into the project folder:

```powershell
git clone https://github.com/ddawdry/ai-in-finance-remake.git
cd ai-in-finance-remake
```

Install the Python packages:

```powershell
python -m pip install -r ml\requirements.txt
```

Install the server packages:

```powershell
cd server
npm install
cd ..
```

Install the frontend packages:

```powershell
cd client
npm install
cd ..
```

## Running the Project

Create results for AAPL:

```powershell
python -m ml.model
```

To create results for every supported stock and cryptocurrency:

```powershell
python -m ml.run_assets
```

You can also choose specific assets:

```powershell
python -m ml.run_assets AAPL MSFT BTC-USD ETH-USD
```

Start the API from the project root:

```powershell
cd server
npm start
```

Open another terminal and start the frontend:

```powershell
cd client
npm run dev
```

Open the local address shown by Vite. This is normally:

```text
http://localhost:5173
```

On Windows, `run_app.bat` can start the model, API, and frontend for you after the packages have been installed.

## Running the Tests

Run the Python tests from the project root:

```powershell
python -m pytest
```

Run the server tests:

```powershell
cd server
npm test
```

Run the frontend tests and lint check:

```powershell
cd client
npm test
npm run lint
```

Check that the frontend production build works:

```powershell
cd client
npm run build
```

## Project Structure

```text
client/        React dashboard
data/          Local market data and generated results
docs/          Longer project notes
ml/            Data preparation, model, evaluation, and backtest code
server/        Express API for dashboard results
tests/         Python tests
```

Downloaded prices and generated model results are ignored by Git. This keeps the repository small and means another user creates their own results locally.

## Model Limits

Market direction is difficult to predict and past results do not guarantee future results. The model only learns from the historical features included in this project. It does not understand company news, wider economic events, or sudden changes in the market.

The backtest is also a simplified historical test. It does not include every cost, delay, or problem that would happen in real trading.

## Project Status

This is an early version that I am continuing to improve. The current focus is building a clear and testable direction model before adding more features.

The first model scope is explained in [docs/model-scope.md](docs/model-scope.md). A longer guide to the data, features, evaluation, and backtest is in [docs/model.md](docs/model.md). The available checks are covered in [docs/testing.md](docs/testing.md).

## Important Note

This project is for learning and portfolio work. It does not provide financial advice and should not be used to make real investment decisions.
