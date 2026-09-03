const express = require("express");
const fs = require("fs/promises");
const path = require("path");

const MODEL_FILE = "model_results.json";
const BACKTEST_FILE = "backtest_results.json";
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

  async function sendResult(res, fileName, label, validator) {
    try {
      const text = await fs.readFile(path.join(resultsDir, fileName), "utf8");
      const result = JSON.parse(text);
      if (!validator(result)) {
        return res.status(500).json({ error: `${label} file is invalid` });
      }
      return res.json(result);
    } catch (error) {
      if (error.code === "ENOENT") {
        return res.status(404).json({
          error: `${label} are not available. Run the model first.`
        });
      }
      return res.status(500).json({ error: `${label} file is invalid` });
    }
  }

  router.get("/model-results", (req, res) => (
    sendResult(res, MODEL_FILE, "Model results", isValidModelResult)
  ));

  router.get("/backtest-results", (req, res) => (
    sendResult(res, BACKTEST_FILE, "Backtest results", isValidBacktestResult)
  ));

  return router;
}

module.exports = { createResultsRouter };
