import os
import json
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

def run_model():
    print("MODEL STARTED")

    tickers = [
        "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"
    ]

    file_path = os.path.join(os.path.dirname(__file__), "marketData.json")

    print("Fetching OHLC data from Yahoo Finance...")

    data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker")

    allrows = []

   # Extract OHLC data
    for tkr in tickers:
        try:
            df_tkr = data[tkr].dropna()

            for dt, row in df_tkr.iterrows():
                allrows.append({
                    "dt": dt,
                    "ticker": tkr,
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"])
                })

        except Exception as e:
            print(f"Skipping {tkr}: {e}")

    df = pd.DataFrame(allrows)

    if df.empty:
        print("No data fetched")
        return

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["ticker", "dt"])

   # Features
    df["ret1"] = df.groupby("ticker")["Close"].pct_change(1)
    df["ret3"] = df.groupby("ticker")["Close"].pct_change(3)
    df["ret7"] = df.groupby("ticker")["Close"].pct_change(7)

    df["ma5"] = df.groupby("ticker")["Close"].rolling(5).mean().reset_index(0, drop=True)
    df["ma10"] = df.groupby("ticker")["Close"].rolling(10).mean().reset_index(0, drop=True)
    df["ma20"] = df.groupby("ticker")["Close"].rolling(20).mean().reset_index(0, drop=True)

    df["momentum"] = df["Close"] - df["ma10"]

    df["volatility"] = df.groupby("ticker")["Close"].pct_change().rolling(5).std().reset_index(0, drop=True)

    df["future"] = df.groupby("ticker")["Close"].shift(-1)
    df["y"] = (df["future"] > df["Close"]).astype(int)

    df = df.dropna()

    features = [
        "ret1", "ret3", "ret7",
        "ma5", "ma10", "ma20",
        "momentum",
        "volatility"
    ]

    # Train models
    print("Training models...")

    models = {}
    accuracies = {}

    for tkr in tickers:
        sub = df[df["ticker"] == tkr]

        if len(sub) < 30:
            print(f"Skipping {tkr}")
            continue

        split = int(len(sub) * 0.8)

        X_train = sub[features][:split]
        y_train = sub["y"][:split]

        X_test = sub[features][split:]
        y_test = sub["y"][split:]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        accuracies[tkr] = round(accuracy, 2)

        print(f"{tkr} accuracy: {accuracy:.2f}")

        models[tkr] = model

    # Predictions and export
    latest = df.groupby("ticker").tail(1).copy()

    export = {}

    for tkr in tickers:
        sub = df[df["ticker"] == tkr]

        if tkr not in models:
            continue

        row = sub.iloc[-1]

        prob = models[tkr].predict_proba(
            pd.DataFrame([row[features]])
        )[0][1]

        # signal
        if prob > 0.65:
            signal = "BUY"
        elif prob < 0.35:
            signal = "SELL"
        else:
            signal = "HOLD"

        hist = sub.tail(15)

        last_price = hist["Close"].iloc[-1]

        predicted = last_price * (1 + (prob - 0.5) * 0.2)

        confidence = abs(prob - 0.5) * 200

        display = tkr.replace("-USD", "")

        # Real candle data
        ohlc_data = [
            {
                "time": d.strftime("%Y-%m-%d"),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c)
            }
            for d, o, h, l, c in zip(
                hist["dt"],
                hist["Open"],
                hist["High"],
                hist["Low"],
                hist["Close"]
            )
        ]

        export[display] = {
            "labels": hist["dt"].dt.strftime("%b %d").tolist(),
            "historical": hist["Close"].round(2).tolist(),
            "ohlc": ohlc_data, 
            "predicted": round(predicted, 2),
            "signal": signal,
            "confidence": round(confidence, 1),
            "sentiment": "Neutral",
            "accuracy": accuracies.get(tkr, None)
        }

    # Save
    with open(file_path, "w") as f:
        json.dump(export, f, indent=2)

    print("Updated marketData.json")


if __name__ == "__main__":
    run_model()
