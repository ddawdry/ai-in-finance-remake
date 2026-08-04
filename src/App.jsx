import { useState, useEffect } from "react";
import axios from "axios";

import Header from "./components/header";
import Sidebar from "./components/Sidebar";
import ChartPanel from "./components/ChartPanel";
import DecisionPanel from "./components/DecisionPanel";
import Login from "./pages/login";

export default function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [market, setMarket] = useState("AAPL");
  const [marketData, setMarketData] = useState(null);
  const [currency, setCurrency] = useState("USD"); // 💱 NEW

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) setLoggedIn(true);
  }, []);

  useEffect(() => {
    if (!loggedIn) return;

    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/market-data");
        setMarketData(res.data);
      } catch (err) {
        console.error(err);
      }
    };

    fetchData();
  }, [loggedIn]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    setLoggedIn(false);
    setMarketData(null);
  };

  if (!loggedIn) {
    return <Login setLoggedIn={setLoggedIn} />;
  }

  if (!marketData) {
    return <div style={{ padding: "20px" }}>Loading...</div>;
  }

  const selectedData = marketData[market];

  return (
    <>
      {/* Pass setCurrency Here */}
      <Header onLogout={handleLogout} setCurrency={setCurrency} />

      <div className="container">
        <Sidebar market={market} setMarket={setMarket} />

        {/* Pass currency Here */}
        <ChartPanel
          market={market}
          data={selectedData}
          currency={currency}
        />

        <DecisionPanel
          market={market}
          data={selectedData}
          currency={currency}
        />
      </div>
    </>
  );
}
