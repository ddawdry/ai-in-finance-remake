# Testing Guide

This project has separate tests for the Python model, Express API, and React frontend. The aim is to catch bad data, future-data leakage, incorrect model results, unsafe ticker input, and broken dashboard formatting before a change is published.

## Before Running Tests

Install the project packages first. The full setup steps are in [setup.md](setup.md).

From the project root, install the Python packages with:

```powershell
python -m pip install -r ml\requirements.txt
```

The server and frontend also need their own npm packages:

```powershell
cd server
npm install
cd ..\client
npm install
cd ..
```

## Python Tests

Run the full Python test suite from the project root:

```powershell
python -m pytest
```

Pytest finds the tests under `tests/ml/` using the settings in `pytest.ini`.

The Python tests cover:

| Test area | What is checked |
| --- | --- |
| Configuration | Tickers, dates, intervals, and random seed values |
| Downloads | Correct `yfinance` settings and clear download errors |
| Data cache | Saving, loading, refreshing, and validating cached prices |
| Validation | Dates, required columns, numeric values, prices, volume, highs, and lows |
| Features | Returns, moving averages, volatility, volume change, and price ranges |
| Dataset | Finished columns, missing rows, date order, and target values |
| Data split | The 80/20 date split and no overlap between training and test rows |
| Classifiers | Predictions, probabilities, settings, metrics, and repeatable results |
| Baselines | Majority-class and market-direction comparison rules |
| Evaluation | Score tables, confusion matrices, and model comparison |
| Walk-forward test | Growing training history and later test blocks |
| Leakage checks | Future prices do not change earlier features or predictions |
| Backtest | Next-day returns, position changes, costs, growth, and risk measurements |
| Multiple assets | Stock and crypto settings, labels, calendars, and ticker checks |
| Result export | Safe CSV and JSON output with valid values and names |

The downloader tests use fake responses. Running the automated tests does not download live market data and does not need existing files under `data/raw/` or `data/results/`.

### Run One Python Test File

During development, it can be quicker to run only the area being changed. For example:

```powershell
python -m pytest tests\ml\test_backtest.py
```

Add `-q` for shorter output:

```powershell
python -m pytest tests\ml\test_features.py -q
```

## Server Tests

Run the server tests with:

```powershell
cd server
npm test
```

These tests start the Express app on a temporary local port. They check:

- The supported asset list
- Loading model results for a selected ticker
- Loading backtest results
- Rejection of unsafe ticker text
- Missing result files
- Invalid JSON and invalid result shapes

The tests create their own temporary result files. They do not use the generated results in the main `data/results/` folder.

## Frontend Tests

Run the frontend tests with:

```powershell
cd client
npm test
```

The current frontend tests check the small data helpers used by the dashboard. This includes:

- Percentage formatting
- Date formatting
- Selecting the latest prediction
- Rejecting missing predictions
- Building the two backtest chart lines
- Rejecting missing chart data

## Frontend Lint Check

Run ESLint from the `client` folder:

```powershell
npm run lint
```

Linting checks the JavaScript and React code for common mistakes. It should finish without errors or warnings.

## Production Build Check

The development server can work even when a production build has a problem. Run this check before publishing a larger frontend change:

```powershell
cd client
npm run build
```

A successful build creates a local `client/dist/` folder. That folder is generated and ignored by Git.

## Dependency Checks

Check the server packages with:

```powershell
cd server
npm audit
```

Check the frontend packages with:

```powershell
cd client
npm audit
```

An audit checks published npm security reports. It needs an internet connection and can sometimes fail because of a network or certificate problem. An audit connection error is different from a reported package vulnerability.

## Manual Dashboard Check

Some parts of the website still need a quick manual check because there is no full browser test suite yet.

After starting the API and frontend, check that:

1. The page loads without an error message.
2. Stocks and crypto appear under separate sidebar headings.
3. Available assets can be selected.
4. The ticker, date, direction, probability, and accuracy values change with the asset.
5. The backtest graph displays both lines.
6. The figures below the graph match the selected asset.
7. The recent direction table contains valid dates and values.
8. The layout remains usable in a narrow browser window.
9. The financial advice warning is visible.

Do not use the manual check to decide whether a model is good. It only checks that the website displays the saved results correctly.

## What a Passing Run Looks Like

Each command should finish with no failed tests. The exact number of tests may increase as the project grows, so the important result is that the failed count stays at zero.

Warnings should still be read. A warning may not fail the command, but it can point to an outdated package or behaviour that will change later.

## Common Problems

### Pytest cannot use the Windows temporary folder

If pytest shows `PermissionError` for a folder under Windows Temp, give it a temporary folder inside the project:

```powershell
python -m pytest --basetemp=.test-temp\pytest
```

The `.test-temp` folder is ignored by Git.

### A command cannot find the project files

Check the current folder with:

```powershell
Get-Location
```

Python tests must be started from the project root. npm commands must be started from either `server` or `client`, depending on the check.

### Frontend tests pass but the page does not load

The frontend helper tests do not start the API. Follow [setup.md](setup.md) and make sure both the API and Vite development server are running.

### Live market data fails

The automated test suite does not require Yahoo Finance. If a real model run fails, check the ticker and internet connection, then try the download again later.

## Current Test Limits

The project does not currently have:

- Full browser automation
- Visual comparison tests
- Performance or load tests
- A live Yahoo Finance test in the normal test suite
- Automatic tests on GitHub for every push

These are useful future improvements. For now, the automated tests cover the model calculations and API result handling, while the dashboard receives a short manual check.
