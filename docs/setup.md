# Setup Guide

This guide explains how to set up and run AI in Finance on a local computer.

## What You Need

Install these before starting:

- Python 3.10 or newer
- Node.js 20.19 or newer
- npm
- Git
- An internet connection for downloading packages and market data

You can check the installed versions with:

```powershell
python --version
node --version
npm --version
git --version
```

## Get the Project

Clone the repository and move into the project folder:

```powershell
git clone https://github.com/ddawdry/ai-in-finance-remake.git
cd ai-in-finance-remake
```

Run the remaining commands from this folder unless the guide says otherwise.

## Install the Python Packages

```powershell
python -m pip install -r ml\requirements.txt
```

Using a virtual environment is optional, but it can keep the project packages separate from other Python projects.

To create and activate one on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt
```

## Install the Server Packages

```powershell
cd server
npm install
cd ..
```

## Install the Frontend Packages

```powershell
cd client
npm install
cd ..
```

## Create Model Results

The downloaded market data and generated results are not stored in Git. You need to create them locally before the dashboard can display them.

Create results for AAPL:

```powershell
python -m ml.model
```

Create results for all supported stocks and cryptocurrencies:

```powershell
python -m ml.run_assets
```

You can also choose a smaller list:

```powershell
python -m ml.run_assets AAPL MSFT BTC-USD ETH-USD
```

The first run downloads daily market data from Yahoo Finance. Later runs use the local cache unless `--refresh` is added:

```powershell
python -m ml.run_assets AAPL MSFT --refresh
```

Model results are saved under `data/results/`. Price data is cached under `data/raw/`. Both folders are ignored by Git apart from their `.gitkeep` files.

## Start the Project Manually

The API and frontend need to run at the same time. Use two terminal windows.

In the first terminal, start the API:

```powershell
cd server
npm start
```

The API normally runs at:

```text
http://127.0.0.1:5000
```

In the second terminal, start the frontend:

```powershell
cd client
npm run dev
```

Vite will print the website address. It is normally:

```text
http://localhost:5173
```

Keep both terminal windows open while using the dashboard. Press `Ctrl+C` in each terminal when you want to stop it.

## Start the Project on Windows

After all packages have been installed, you can run:

```powershell
.\run_app.bat
```

The batch file creates the first AAPL result if needed, opens the API and frontend in separate windows, and then opens the website.

## Run the Checks

Run all Python tests from the project root:

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

Create a production frontend build:

```powershell
cd client
npm run build
```

## Common Problems

### The page says connection refused

Check that the frontend terminal is still open and that Vite says it is ready. Restart it with:

```powershell
cd client
npm run dev
```

### The page says results are not available

Make sure the API is running and create at least one model result:

```powershell
python -m ml.model
```

Then restart the API and refresh the page.

### An asset is greyed out

The dashboard only enables assets that have local model and backtest results. Run the model for that asset. For example:

```powershell
python -m ml.run_assets TSLA BTC-USD
```

### Yahoo Finance returns no data

Check the internet connection and ticker name, then try again later. Yahoo Finance may sometimes reject or delay a request. Existing cached data can still be used without `--refresh`.

### PowerShell blocks the virtual environment

You can either use Python without a virtual environment or allow the activation script for the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This policy change only lasts until that PowerShell window is closed.
