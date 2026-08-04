# AI in Finance

A university group project that explores the use of machine learning with stock and cryptocurrency market data.

## About the Project

The application uses recent market data to create simple buy, hold, or sell signals. It displays the results in a React dashboard with price charts, confidence values, currency options, and a basic investment simulator.

The original project was built by a group of four students. The work was split between:

- AI and machine learning
- Frontend design
- Finding reliable market data
- Testing, data input, and data sanitisation

My main role was testing, data input, and data sanitisation. I am now rebuilding and improving the project myself so I can learn more about every part of it.

## Current Features

- Stock and cryptocurrency market data from Yahoo Finance
- A basic Random Forest machine learning model
- Buy, hold, and sell signals
- Historical price charts
- Currency display options
- A simple investment simulator
- Local user login with MongoDB

## Setup

You will need Node.js, Python 3, and MongoDB.

Install the frontend packages:

```powershell
npm install
```

Install the backend packages:

```powershell
cd server
npm install
cd ..
```

Install the Python packages:

```powershell
pip install pandas yfinance scikit-learn
```

Create the local environment file:

```powershell
Copy-Item server\.env.example server\.env
```

Update `server/.env` with your own MongoDB connection and JWT secret. Do not commit this file.

## Running the Project

Update the market data:

```powershell
python src\ML.py
```

Start the backend:

```powershell
cd server
node server.js
```

Start the frontend in another terminal:

```powershell
npm run dev
```

## Project Status

This is an early version that I plan to improve over time. The setup process, design, testing, data handling, and machine learning model all need more work.

## Tools

- Python
- pandas, yfinance, and scikit-learn
- React and Vite
- Node.js and Express
- MongoDB
- Chart.js

## Important Note

This project is for learning and portfolio work. Its predictions are not financial advice and should not be used to make real investment decisions.
