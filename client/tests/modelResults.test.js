import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBacktestChartData,
  formatDate,
  formatPercent,
  getLatestPrediction,
} from "../src/modelResults.js";

test("formats model values as percentages", () => {
  assert.equal(formatPercent(0.49905), "49.9%");
  assert.equal(formatPercent(0.6), "60.0%");
  assert.equal(formatPercent(undefined), "N/A");
});

test("formats an export date for the dashboard", () => {
  assert.equal(formatDate("2026-08-10"), "10 Aug 2026");
  assert.equal(formatDate("bad-date"), "Unknown date");
});

test("returns the newest prediction", () => {
  const result = {
    predictions: [{ date: "2025-01-01" }, { date: "2025-01-02" }],
  };

  assert.equal(getLatestPrediction(result).date, "2025-01-02");
});

test("rejects missing predictions", () => {
  assert.throws(
    () => getLatestPrediction({ predictions: [] }),
    /do not contain predictions/
  );
});

test("builds percentage lines for the backtest chart", () => {
  const chart = buildBacktestChartData([
    { date: "2025-01-01", strategy_growth: 1.1, buy_hold_growth: 0.95 },
    { date: "2025-01-02", strategy_growth: 1.2, buy_hold_growth: 1.05 },
  ]);

  assert.deepEqual(chart.labels, ["01 Jan 2025", "02 Jan 2025"]);
  assert.deepEqual(
    chart.datasets[0].data.map((value) => Number(value.toFixed(1))),
    [10, 20]
  );
  assert.deepEqual(
    chart.datasets[1].data.map((value) => Number(value.toFixed(1))),
    [-5, 5]
  );
});

test("rejects missing backtest chart rows", () => {
  assert.throws(
    () => buildBacktestChartData([]),
    /do not contain chart data/
  );
});
