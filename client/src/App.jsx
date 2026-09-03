import { useEffect, useState } from "react";
import axios from "axios";

import "./App.css";
import Header from "./components/header";
import BacktestPanel from "./components/BacktestPanel";
import DecisionPanel from "./components/DecisionPanel";
import Login from "./pages/login";
import {
  formatDate,
  formatPercent,
  getLatestPrediction,
} from "./modelResults";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [loggedIn, setLoggedIn] = useState(() => (
    Boolean(localStorage.getItem("token"))
  ));
  const [modelResult, setModelResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);
  const [status, setStatus] = useState("loading");
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    if (!loggedIn) return undefined;

    let cancelled = false;

    Promise.all([
      axios.get(`${API_URL}/api/model-results`),
      axios.get(`${API_URL}/api/backtest-results`),
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
  }, [loggedIn, reloadCount]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    setLoggedIn(false);
    setModelResult(null);
    setBacktestResult(null);
  };

  if (!loggedIn) {
    return <Login setLoggedIn={setLoggedIn} />;
  }

  return (
    <div className="app-shell">
      <Header onLogout={handleLogout} />

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
        />
      )}
    </div>
  );
}

function DashboardContent({ result, backtest }) {
  const latest = getLatestPrediction(result);
  const recent = result.predictions.slice(-10).reverse();

  return (
    <main className="dashboard">
      <div className="dashboard-heading">
        <div>
          <p className="asset-type">{result.asset_type}</p>
          <h1>{result.ticker}</h1>
        </div>
        <div className="as-of">
          <span>Latest test date</span>
          <strong>{formatDate(latest.date)}</strong>
        </div>
      </div>

      <DecisionPanel result={result} prediction={latest} />
      <BacktestPanel result={backtest} />

      <section className="history-section" aria-labelledby="history-title">
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
  );
}
