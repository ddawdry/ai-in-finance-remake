import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

import {
  buildBacktestChartData,
  formatPercent,
} from "../modelResults";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

export default function BacktestPanel({ result }) {
  const chartData = buildBacktestChartData(result.curve);
  const strategyReturn = result.strategy_after_costs.total_return;
  const buyHoldReturn = result.buy_hold.total_return;
  const strategyWon = strategyReturn > buyHoldReturn;
  const difference = Math.abs(strategyReturn - buyHoldReturn);
  const leader = strategyWon ? "The direction strategy" : "Buy and hold";

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: "index" },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#8f928f", maxTicksLimit: 8 },
      },
      y: {
        grid: { color: "#303230" },
        ticks: {
          color: "#8f928f",
          callback: (value) => `${value}%`,
        },
      },
    },
    plugins: {
      legend: { labels: { color: "#d9dad7", usePointStyle: true } },
      tooltip: {
        callbacks: {
          label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%`,
        },
      },
    },
  };

  return (
    <section className="backtest-section" aria-labelledby="backtest-title">
      <div className="section-heading">
        <div>
          <h2 id="backtest-title">Historical backtest</h2>
          <p>Strategy after costs compared with buy and hold</p>
        </div>
        <span>Historical test only. Not financial advice.</span>
      </div>

      <div className="backtest-layout">
        <div className="backtest-chart">
          <Line data={chartData} options={options} />
        </div>

        <div className="backtest-summary">
          <div className="summary-row">
            <span>Strategy return after costs</span>
            <strong>{formatPercent(strategyReturn)}</strong>
          </div>
          <div className="summary-row">
            <span>Buy and hold return</span>
            <strong>{formatPercent(buyHoldReturn)}</strong>
          </div>
          <div className="summary-row">
            <span>Largest strategy drop</span>
            <strong>
              {formatPercent(Math.abs(result.strategy_after_costs.maximum_drawdown))}
            </strong>
          </div>
          <div className="summary-row">
            <span>Trading cost</span>
            <strong>{formatPercent(result.assumptions.trading_cost)}</strong>
          </div>
          <p className="backtest-outcome">
            {leader} finished {formatPercent(difference)} ahead.
          </p>
        </div>
      </div>
    </section>
  );
}
