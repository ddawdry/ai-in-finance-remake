export default function Header({ onLogout, setCurrency }) {
  return (
    <header style={styles.header}>
      <h1 style={styles.logo}>AI Finance</h1>

      <div style={styles.right}>
        {/* Currency Dropdown */}
        <select
          onChange={(e) => setCurrency(e.target.value)}
          style={styles.dropdown}
        >
          <option value="USD">USD ($)</option>
          <option value="GBP">GBP (£)</option>
          <option value="EUR">EUR (€)</option>
          <option value="JPY">JPY (¥)</option>
        </select>

        <span style={styles.disclaimer}>Not Financial Advice</span>

        <button style={styles.logout} onClick={onLogout}>
          Logout
        </button>
      </div>
    </header>
  );
}

const styles = {
  header: {
    height: "60px",
    backgroundColor: "#020617",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 25px",
    borderBottom: "1px solid #1e293b",
  },
  logo: {
    margin: 0,
    color: "#38bdf8",
  },
  right: {
    display: "flex",
    alignItems: "center",
    gap: "15px",
  },
  dropdown: {
    backgroundColor: "#1e293b",
    color: "#e5e7eb",
    border: "none",
    padding: "6px",
    borderRadius: "6px",
    cursor: "pointer",
  },
  disclaimer: {
    fontSize: "0.9rem",
    color: "#facc15",
  },
  logout: {
    padding: "6px 12px",
    backgroundColor: "#ef4444",
    border: "none",
    borderRadius: "6px",
    color: "white",
    cursor: "pointer",
  },
};
