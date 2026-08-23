import assert from "node:assert/strict";
import test from "node:test";

import {
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
