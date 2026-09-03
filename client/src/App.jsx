import { useEffect, useState } from "react";
import axios from "axios";

import "./App.css";
import Header from "./components/header";
import BacktestPanel from "./components/BacktestPanel";
import DecisionPanel from "./components/DecisionPanel";
import {
  formatDate,
  formatPercent,
  getLatestPrediction,
} from "./modelResults";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [modelResult, setModelResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);
  const [assets, setAssets] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [status, setStatus] = useState("loading");
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    let cancelled = false;

    axios.get(`${API_URL}/api/assets`)
      .then((response) => {
        if (!cancelled) {
          const availableAssets = response.data.assets;
          setAssets(availableAssets);
          const readyAssets = availableAssets.filter((asset) => asset.available);
          if (!readyAssets.length) {
            setStatus("error");
            return;
          }
          setSelectedTicker((current) => (
            readyAssets.some((asset) => asset.ticker === current)
              ? current
              : (readyAssets.find((asset) => asset.ticker === "AAPL") ||
                readyAssets[0]).ticker
          ));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelResult(null);
          setBacktestResult(null);
          setStatus("error");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [reloadCount]);

  useEffect(() => {
    if (!selectedTicker) return undefined;

    let cancelled = false;
    const ticker = encodeURIComponent(selectedTicker);
    Promise.all([
      axios.get(`${API_URL}/api/model-results?ticker=${ticker}`),
      axios.get(`${API_URL}/api/backtest-results?ticker=${ticker}`),
    ])
      .then(([modelResponse, backtestResponse]) => {
        if (!cancelled) {
          setModelResult(modelResponse.data);
          setBacktestResult(backtestResponse.data);
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelResult(null);
          setBacktestResult(null);
          setStatus("error");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTicker, reloadCount]);

  const handleSelectTicker = (ticker) => {
    if (ticker === selectedTicker) return;
    setStatus("loading");
    setSelectedTicker(ticker);
  };

  return (
    <div className="app-shell">
      <Header />

      {status === "loading" && (
        <main className="state-page" aria-live="polite">
          <div className="loading-mark" />
          <p>Loading model results...</p>
        </main>
      )}

      {status === "error" && (
        <main className="state-page" role="alert">
          <h2>Results are not available</h2>
          <p>Run the model and make sure the server is running.</p>
          <button
            type="button"
            onClick={() => {
              setStatus("loading");
              setReloadCount((count) => count + 1);
            }}
          >
            Try again
          </button>
        </main>
      )}

      {status === "ready" && modelResult && backtestResult && (
        <DashboardContent
          result={modelResult}
          backtest={backtestResult}
          assets={assets}
          selectedTicker={selectedTicker}
          onSelectTicker={handleSelectTicker}
        />
      )}
    </div>
  );
}

function DashboardContent({
  result,
  backtest,
  assets,
  selectedTicker,
  onSelectTicker,
}) {
  const latest = getLatestPrediction(result);
  const recent = result.predictions.slice(-10).reverse();

  return (
    <div className="dashboard-layout">
      <aside className="market-sidebar" aria-label="Dashboard navigation">
        <div className="sidebar-section asset-section">
          <p className="sidebar-title">Markets</p>
          <p className="sidebar-group">Stocks</p>
          {assets.filter((asset) => asset.asset_type === "stock").map((asset) => (
            <button
              type="button"
              className={`ticker-link ${
                asset.ticker === selectedTicker ? "active" : ""
              }`}
              onClick={() => onSelectTicker(asset.ticker)}
              disabled={!asset.available}
              title={asset.available ? asset.ticker : "Run the model to add results"}
              key={asset.ticker}
            >
              {asset.ticker}
            </button>
          ))}

          <p className="sidebar-group">Crypto</p>
          {assets.filter((asset) => asset.asset_type === "crypto").map((asset) => (
            <button
              type="button"
              className={`ticker-link ${
                asset.ticker === selectedTicker ? "active" : ""
              }`}
              onClick={() => onSelectTicker(asset.ticker)}
              disabled={!asset.available}
              title={asset.available ? asset.ticker : "Run the model to add results"}
              key={asset.ticker}
            >
              {asset.ticker.replace("-USD", "")}
            </button>
          ))}
        </div>

        <nav className="sidebar-section" aria-label="Page sections">
          <p className="sidebar-title">View</p>
          <a href="#overview">Overview</a>
          <a href="#backtest">Backtest</a>
          <a href="#history">History</a>
        </nav>

        <p className="sidebar-note">
          This tool shows model test results. It does not place trades.
        </p>
      </aside>

      <main className="dashboard">
        <div className="dashboard-heading" id="overview">
          <div>
            <p className="asset-type">{result.asset_type} / daily direction</p>
            <h1>{result.ticker} market model</h1>
          </div>
          <div className="as-of">
            <span>Latest test date</span>
            <strong>{formatDate(latest.date)}</strong>
          </div>
        </div>

        <DecisionPanel result={result} prediction={latest} />

        <div id="backtest">
          <BacktestPanel result={backtest} />
        </div>

        <section
          className="history-section"
          id="history"
          aria-labelledby="history-title"
        >
          <div className="section-heading">
            <h2 id="history-title">Recent direction results</h2>
            <span>{result.model.replaceAll("_", " ")}</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Actual</th>
                  <th>Predicted</th>
                  <th>Up probability</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((row) => (
                  <tr key={row.date}>
                    <td>{formatDate(row.date)}</td>
                    <td className={`direction-text ${row.actual_direction}`}>
                      {row.actual_direction}
                    </td>
                    <td className={`direction-text ${row.predicted_direction}`}>
                      {row.predicted_direction}
                    </td>
                    <td>{formatPercent(row.up_probability)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
