import { formatPercent } from "../modelResults";

export default function DecisionPanel({ result, prediction }) {
  const direction = prediction.predicted_direction;
  const directionLabel = direction === "up" ? "Up" : "Down";

  return (
    <section className="result-overview" aria-label="Latest model result">
      <div className={`direction-panel ${direction}`}>
        <span className="panel-label">Predicted direction</span>
        <strong>{directionLabel}</strong>
        <span className="panel-detail">Next trading day</span>
      </div>

      <div className="metric-panel">
        <span className="panel-label">Chance of an up day</span>
        <strong>{formatPercent(prediction.up_probability)}</strong>
        <div className="probability-track" aria-hidden="true">
          <span style={{ width: formatPercent(prediction.up_probability) }} />
        </div>
      </div>

      <div className="metric-panel">
        <span className="panel-label">Model accuracy</span>
        <strong>{formatPercent(result.metrics.accuracy)}</strong>
        <span className="panel-detail">Walk-forward test</span>
      </div>

      <div className="metric-panel">
        <span className="panel-label">Baseline accuracy</span>
        <strong>{formatPercent(result.baseline.metrics.accuracy)}</strong>
        <span className="panel-detail">Majority class</span>
      </div>
    </section>
  );
}
