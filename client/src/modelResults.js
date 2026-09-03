export function formatPercent(value) {
  if (!Number.isFinite(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDate(value) {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return "Unknown date";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function getLatestPrediction(result) {
  if (!result || !Array.isArray(result.predictions) || !result.predictions.length) {
    throw new Error("Model results do not contain predictions.");
  }
  return result.predictions[result.predictions.length - 1];
}

export function buildBacktestChartData(curve) {
  if (!Array.isArray(curve) || !curve.length) {
    throw new Error("Backtest results do not contain chart data.");
  }

  return {
    labels: curve.map((row) => formatDate(row.date)),
    datasets: [
      {
        label: "Direction strategy",
        data: curve.map((row) => (row.strategy_growth - 1) * 100),
        borderColor: "#71ce91",
        backgroundColor: "#71ce91",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
      },
      {
        label: "Buy and hold",
        data: curve.map((row) => (row.buy_hold_growth - 1) * 100),
        borderColor: "#e7b84b",
        backgroundColor: "#e7b84b",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
      },
    ],
  };
}
