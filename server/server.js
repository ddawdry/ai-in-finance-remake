const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const fs = require("fs");
const path = require("path");

const authRoutes = require("./routes/auth");
const authMiddleware = require("./middleware/authMiddleware");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Routes

// Auth routes
app.use("/api/auth", authRoutes);

// Protected test route
app.get("/api/protected", authMiddleware, (req, res) => {
  res.json({ message: "You accessed a protected route!" });
});

// Market Data API
app.get("/api/market-data", (req, res) => {
  const filePath = path.join(__dirname, "../src/marketData.json");

  fs.readFile(filePath, "utf8", (err, data) => {
    if (err) {
      console.error("Error reading marketData.json:", err);
      return res.status(500).json({ error: "Failed to load market data" });
    }

    try {
      const parsed = JSON.parse(data);
      res.json(parsed);
    } catch (parseErr) {
      console.error("JSON parse error:", parseErr);
      res.status(500).json({ error: "Invalid JSON format" });
    }
  });
});

// Database
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.log("MongoDB error:", err));

// Server
const PORT = 5000;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
