import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

export default function ChartPanel({ market, data, currency }) {
  if (!data) {
    return <main style={{ padding: "20px" }}>Loading...</main>;
  }

  const rates = {
    USD: 1,
    GBP: 0.79,
    EUR: 0.92,
    JPY: 155,
  };

  const symbols = {
    USD: "$",
    GBP: "£",
    EUR: "€",
    JPY: "¥",
  };

  const rate = rates[currency] || 1;

  const convertedHistorical = data.historical.map(p => p * rate);
  const convertedPrediction = data.predicted * rate;

  const chartData = {
    labels: [...data.labels, "Prediction"],
    datasets: [
      {
        label: `Historical (${currency})`,
        data: [...convertedHistorical, null],
        borderColor: "#38bdf8",
        tension: 0.3,
      },
      {
        label: "AI Forecast",
        data: [
          ...Array(convertedHistorical.length - 1).fill(null),
          convertedHistorical[convertedHistorical.length - 1],
          convertedPrediction,
        ],
        borderColor: data.signal === "BUY" ? "#22c55e" : "#ef4444",
        borderDash: [5, 5],
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        ticks: { color: "#94a3b8" },
        grid: { color: "#1e293b" },
      },
      y: {
        ticks: {
          color: "#94a3b8",
          callback: (value) => `${symbols[currency]}${value.toFixed(2)}`,
        },
        grid: { color: "#1e293b" },
      },
    },
    plugins: {
      legend: {
        labels: { color: "#e5e7eb" },
      },
      tooltip: {
        callbacks: {
          label: (context) =>
            `${symbols[currency]}${context.raw.toFixed(2)}`,
        },
      },
    },
  };

  return (
    <main style={styles.container}>
      <h2>
        {market} / {currency} ({symbols[currency]})
      </h2>

      <div style={styles.chartWrapper}>
        <Line data={chartData} options={options} />
      </div>

      <p style={styles.footer}>
        Sentiment: {data.sentiment}
      </p>
    </main>
  );
}

const styles = {
  container: {
    padding: "20px",
    backgroundColor: "#0f172a",
    flex: 1,
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  chartWrapper: {
    flex: 1,
    position: "relative",
  },
  footer: {
    marginTop: "10px",
    color: "#94a3b8",
  },
};