export default function Sidebar({ market, setMarket }) {

  const stocks = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"];
  const crypto = ["BTC", "ETH", "SOL", "ADA", "DOGE"];

  return (
    <aside style={styles.sidebar}>
      <h2>Markets</h2>

      {/* Stocks */}
      <h3 style={styles.section}>Stocks</h3>
      <ul style={styles.list}>
        {stocks.map((m) => (
          <li
            key={m}
            onClick={() => setMarket(m)}
            style={{
              ...styles.item,
              ...(market === m ? styles.active : {}),
            }}
          >
            {m}
          </li>
        ))}
      </ul>

      {/* Crypto */}
      <h3 style={styles.section}>Crypto</h3>
      <ul style={styles.list}>
        {crypto.map((m) => (
          <li
            key={m}
            onClick={() => setMarket(m)}
            style={{
              ...styles.item,
              ...(market === m ? styles.active : {}),
            }}
          >
            {m}
          </li>
        ))}
      </ul>

    </aside>
  );
}

const styles = {
  sidebar: {
    backgroundColor: "#020617",
    padding: "20px",
    borderRight: "1px solid #1e293b"
  },
  list: {
    listStyle: "none",
    padding: 0,
    margin: 0
  },
  item: {
    padding: "8px 0",
    cursor: "pointer",
    color: "#94a3b8"
  },
  active: {
    color: "#38bdf8",
    fontWeight: "bold"
  },
  section: {
    marginTop: "15px",
    color: "#e5e7eb",
    fontSize: "0.9rem"
  }
};
