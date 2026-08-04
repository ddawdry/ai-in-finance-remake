import { useState } from "react";

export default function Simulator({ data, currency }) {
  const [amount, setAmount] = useState("");

  // same rates as chart
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

  // Convert input → USD → apply prediction → convert back
  const calculateReturn = () => {
    if (!amount || !data) return 0;

    const input = parseFloat(amount);

    const usdValue = input / rate; // convert to USD
    const predictedUSD =
      usdValue * (data.predicted / data.historical[data.historical.length - 1]);

    return predictedUSD * rate; // convert back to selected currency
  };

  const result = calculateReturn();

  return (
    <div style={styles.container}>
      <h3>Investment Simulator</h3>

      <input
        type="number"
        placeholder={`Enter amount (${symbols[currency]})`}
        value={amount}
        onChange={(e) => setAmount(e.target.value)}
        style={styles.input}
      />

      {amount && (
        <p style={styles.result}>
          Estimated Value: {symbols[currency]}
          {result.toFixed(2)}
        </p>
      )}
    </div>
  );
}

const styles = {
  container: {
    marginTop: "20px",
    padding: "15px",
    border: "1px solid #1e293b",
    borderRadius: "10px",
  },
  input: {
    width: "100%",
    padding: "10px",
    borderRadius: "6px",
    border: "1px solid #1e293b",
    backgroundColor: "#020617",
    color: "white",
  },
  result: {
    marginTop: "10px",
    color: "#22c55e",
  },
};
