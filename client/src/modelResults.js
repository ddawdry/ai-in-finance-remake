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
