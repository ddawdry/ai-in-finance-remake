const express = require("express");
const fs = require("fs/promises");
const path = require("path");

const MODEL_FILE = "model_results.json";
const BACKTEST_FILE = "backtest_results.json";
const SAFE_TICKER = /^[A-Z0-9._-]+$/;
const SUPPORTED_ASSETS = [
  { ticker: "AAPL", asset_type: "stock" },
  { ticker: "MSFT", asset_type: "stock" },
  { ticker: "TSLA", asset_type: "stock" },
  { ticker: "GOOGL", asset_type: "stock" },
  { ticker: "AMZN", asset_type: "stock" },
  { ticker: "BTC-USD", asset_type: "crypto" },
  { ticker: "ETH-USD", asset_type: "crypto" },
  { ticker: "SOL-USD", asset_type: "crypto" },
  { ticker: "ADA-USD", asset_type: "crypto" },
  { ticker: "DOGE-USD", asset_type: "crypto" }
];
const MODEL_METRICS = ["accuracy", "precision", "recall", "f1"];
const PERFORMANCE_METRICS = [
  "total_return",
  "annualised_return",
  "annualised_volatility",
  "maximum_drawdown",
  "sharpe_ratio"
];

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasNumbers(value, names) {
  return isObject(value) && names.every((name) => Number.isFinite(value[name]));
}

function isValidModelResult(value) {
  if (
    !isObject(value) ||
    typeof value.ticker !== "string" ||
    !["stock", "crypto"].includes(value.asset_type) ||
    !hasNumbers(value.metrics, MODEL_METRICS) ||
    !isObject(value.baseline) ||
    value.baseline.name !== "majority_class" ||
    !hasNumbers(value.baseline.metrics, MODEL_METRICS) ||
    !Array.isArray(value.predictions)
  ) {
    return false;
  }

  return value.predictions.every((row) => (
    isObject(row) &&
    typeof row.date === "string" &&
    row.ticker === value.ticker &&
    row.asset_type === value.asset_type &&
    ["up", "down"].includes(row.actual_direction) &&
    ["up", "down"].includes(row.predicted_direction) &&
    Number.isFinite(row.up_probability) &&
    row.up_probability >= 0 &&
    row.up_probability <= 1
  ));
}

function isValidBacktestResult(value) {
  return (
    isObject(value) &&
    typeof value.ticker === "string" &&
    ["stock", "crypto"].includes(value.asset_type) &&
    value.strategy === "up_prediction_only" &&
    isObject(value.assumptions) &&
    Number.isFinite(value.assumptions.trading_cost) &&
    Number.isInteger(value.assumptions.trading_days_per_year) &&
    Number.isFinite(value.assumptions.risk_free_rate) &&
    Array.isArray(value.curve) &&
    value.curve.length > 0 &&
    value.curve.every((row) => (
      isObject(row) &&
      typeof row.date === "string" &&
      Number.isFinite(row.strategy_growth) &&
      Number.isFinite(row.buy_hold_growth)
    )) &&
    hasNumbers(value.strategy_before_costs, PERFORMANCE_METRICS) &&
    hasNumbers(value.strategy_after_costs, PERFORMANCE_METRICS) &&
    hasNumbers(value.buy_hold, PERFORMANCE_METRICS)
  );
}

function createResultsRouter({ resultsDir }) {
  const router = express.Router();

  function readTicker(req) {
    if (req.query.ticker === undefined) return null;
    const ticker = String(req.query.ticker).trim().toUpperCase();
    return SAFE_TICKER.test(ticker) && ![".", ".."].includes(ticker)
      ? ticker
      : undefined;
  }

  function resultPath(fileName, ticker) {
    return ticker
      ? path.join(resultsDir, ticker, fileName)
      : path.join(resultsDir, fileName);
  }

  async function sendResult(res, fileName, label, validator, ticker = null) {
    const locations = ticker ? [ticker, null] : [null];

    for (const location of locations) {
      try {
        const text = await fs.readFile(resultPath(fileName, location), "utf8");
        const result = JSON.parse(text);
        if (!validator(result) || (ticker && result.ticker !== ticker)) {
          return res.status(500).json({ error: `${label} file is invalid` });
        }
        return res.json(result);
      } catch (error) {
        if (error.code !== "ENOENT") {
          return res.status(500).json({ error: `${label} file is invalid` });
        }
      }
    }

    return res.status(404).json({
      error: `${label} are not available. Run the model first.`
    });
  }

  router.get("/assets", async (_req, res) => {
    const available = new Set();

    try {
      const entries = await fs.readdir(resultsDir, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory() || !SAFE_TICKER.test(entry.name)) continue;

        try {
          const [modelText, backtestText] = await Promise.all([
            fs.readFile(resultPath(MODEL_FILE, entry.name), "utf8"),
            fs.readFile(resultPath(BACKTEST_FILE, entry.name), "utf8")
          ]);
          const model = JSON.parse(modelText);
          const backtest = JSON.parse(backtestText);
          if (
            isValidModelResult(model) &&
            isValidBacktestResult(backtest) &&
            model.ticker === entry.name &&
            backtest.ticker === entry.name &&
            model.asset_type === backtest.asset_type
          ) {
            available.add(model.ticker);
          }
        } catch {
          // Ignore incomplete asset folders.
        }
      }

      if (!available.size) {
        try {
          const [modelText, backtestText] = await Promise.all([
            fs.readFile(resultPath(MODEL_FILE, null), "utf8"),
            fs.readFile(resultPath(BACKTEST_FILE, null), "utf8")
          ]);
          const model = JSON.parse(modelText);
          const backtest = JSON.parse(backtestText);
          if (
            isValidModelResult(model) &&
            isValidBacktestResult(backtest) &&
            model.ticker === backtest.ticker &&
            model.asset_type === backtest.asset_type
          ) {
            available.add(model.ticker);
          }
        } catch {
          // The empty list tells the client that no results exist yet.
        }
      }

      const assets = SUPPORTED_ASSETS.map((asset) => ({
        ...asset,
        available: available.has(asset.ticker)
      }));
      return res.json({ assets });
    } catch (error) {
      if (error.code === "ENOENT") {
        return res.json({
          assets: SUPPORTED_ASSETS.map((asset) => ({
            ...asset,
            available: false
          }))
        });
      }
      return res.status(500).json({ error: "Asset list is not available" });
    }
  });

  router.get("/model-results", (req, res) => {
    const ticker = readTicker(req);
    if (ticker === undefined) {
      return res.status(400).json({ error: "Ticker is invalid" });
    }
    return sendResult(res, MODEL_FILE, "Model results", isValidModelResult, ticker);
  });

  router.get("/backtest-results", (req, res) => {
    const ticker = readTicker(req);
    if (ticker === undefined) {
      return res.status(400).json({ error: "Ticker is invalid" });
    }
    return sendResult(
      res,
      BACKTEST_FILE,
      "Backtest results",
      isValidBacktestResult,
      ticker
    );
  });

  return router;
}

module.exports = { createResultsRouter };
