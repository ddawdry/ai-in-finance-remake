import { useState } from "react";
import axios from "axios";

export default function Login({ setLoggedIn }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    try {
      const res = await axios.post(
        "http://localhost:5000/api/auth/login",
        { email, password }
      );

      localStorage.setItem("token", res.data.token);
      setLoggedIn(true);

    } catch {
      alert("Login failed");
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>

        <h1 style={styles.title}>AI Finance</h1>
        <p style={styles.subtitle}>Smart Market Predictions</p>

        <input
          style={styles.input}
          type="email"
          placeholder="Email"
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          style={styles.input}
          type="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
        />

        <button style={styles.button} onClick={handleLogin}>
          Login
        </button>

      </div>
    </div>
  );
}

const styles = {
  page: {
    height: "100vh",
    background: "linear-gradient(135deg, #020617, #0f172a)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  card: {
    backgroundColor: "#020617",
    padding: "40px",
    borderRadius: "12px",
    width: "320px",
    boxShadow: "0 0 30px rgba(56,189,248,0.2)",
    border: "1px solid #1e293b",
    display: "flex",
    flexDirection: "column",
  },
  title: {
    color: "#38bdf8",
    marginBottom: "5px",
    textAlign: "center",
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: "0.9rem",
    marginBottom: "25px",
    textAlign: "center",
  },
  input: {
    padding: "10px",
    marginBottom: "15px",
    borderRadius: "6px",
    border: "1px solid #1e293b",
    backgroundColor: "#0f172a",
    color: "white",
  },
  button: {
    padding: "10px",
    backgroundColor: "#38bdf8",
    border: "none",
    borderRadius: "6px",
    fontWeight: "bold",
    cursor: "pointer",
    color: "#020617",
    transition: "0.2s",
  },
};