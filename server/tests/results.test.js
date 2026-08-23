const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");
const test = require("node:test");

const { createApp } = require("../app");

const resultsDir = path.join(__dirname, "../.test-temp/results");

function modelResult() {
  return {
    ticker: "AAPL",
    asset_type: "stock",
    model: "random_forest_walk_forward",
    prediction_rows: 1,
    start_date: "2025-01-02",
    end_date: "2025-01-02",
    metrics: {
      accuracy: 0.5,
      precision: 0.6,
      recall: 0.7,
      f1: 0.65
    },
    baseline: {
      name: "majority_class",
      metrics: {
        accuracy: 0.55,
        precision: 0.55,
        recall: 1,
        f1: 0.71
      }
    },
    predictions: [{
      date: "2025-01-02",
      ticker: "AAPL",
      asset_type: "stock",
      actual_direction: "up",
      predicted_direction: "down",
      up_probability: 0.4
    }]
  };
}

function performanceMetrics(totalReturn) {
  return {
    total_return: totalReturn,
    annualised_return: 0.08,
    annualised_volatility: 0.2,
    maximum_drawdown: -0.15,
    sharpe_ratio: 0.5
  };
}

function backtestResult() {
  return {
    ticker: "AAPL",
    asset_type: "stock",
    strategy: "up_prediction_only",
    assumptions: {
      trading_cost: 0.001,
      trading_days_per_year: 252,
      risk_free_rate: 0
    },
    strategy_before_costs: performanceMetrics(0.1),
    strategy_after_costs: performanceMetrics(0.07),
    buy_hold: performanceMetrics(0.12)
  };
}

async function writeJson(fileName, value) {
  await fs.mkdir(resultsDir, { recursive: true });
  await fs.writeFile(
    path.join(resultsDir, fileName),
    JSON.stringify(value),
    "utf8"
  );
}

async function removeResults() {
  await fs.rm(resultsDir, { recursive: true, force: true });
}

function requestJson(server, route) {
  const { port } = server.address();
  return new Promise((resolve, reject) => {
    http.get({ hostname: "127.0.0.1", port, path: route }, (response) => {
      let text = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        text += chunk;
      });
      response.on("end", () => {
        resolve({ status: response.statusCode, body: JSON.parse(text) });
      });
    }).on("error", reject);
  });
}

async function withServer(callback) {
  const server = createApp({ resultsDir }).listen(0, "127.0.0.1");
  try {
    await new Promise((resolve) => server.once("listening", resolve));
    return await callback(server);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test.beforeEach(removeResults);
test.after(removeResults);

test("returns valid model results", async () => {
  const expected = modelResult();
  await writeJson("model_results.json", expected);

  const response = await withServer((server) => (
    requestJson(server, "/api/model-results")
  ));

  assert.equal(response.status, 200);
  assert.deepEqual(response.body, expected);
});

test("returns 404 when model results are missing", async () => {
  const response = await withServer((server) => (
    requestJson(server, "/api/model-results")
  ));

  assert.equal(response.status, 404);
  assert.match(response.body.error, /not available/i);
});

test("returns 500 when model results are invalid", async () => {
  await writeJson("model_results.json", { ticker: "AAPL" });

  const response = await withServer((server) => (
    requestJson(server, "/api/model-results")
  ));

  assert.equal(response.status, 500);
  assert.match(response.body.error, /invalid/i);
});

test("returns valid backtest results", async () => {
  const expected = backtestResult();
  await writeJson("backtest_results.json", expected);

  const response = await withServer((server) => (
    requestJson(server, "/api/backtest-results")
  ));

  assert.equal(response.status, 200);
  assert.deepEqual(response.body, expected);
});

test("returns 404 when backtest results are missing", async () => {
  const response = await withServer((server) => (
    requestJson(server, "/api/backtest-results")
  ));

  assert.equal(response.status, 404);
  assert.match(response.body.error, /not available/i);
});

test("returns 500 when backtest results are invalid", async () => {
  await fs.mkdir(resultsDir, { recursive: true });
  await fs.writeFile(
    path.join(resultsDir, "backtest_results.json"),
    "not-json",
    "utf8"
  );

  const response = await withServer((server) => (
    requestJson(server, "/api/backtest-results")
  ));

  assert.equal(response.status, 500);
  assert.match(response.body.error, /invalid/i);
});
