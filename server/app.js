const express = require("express");
const cors = require("cors");
const path = require("path");

const authRoutes = require("./routes/auth");
const authMiddleware = require("./middleware/authMiddleware");
const { createResultsRouter } = require("./routes/results");

function createApp(options = {}) {
  const resultsDir = options.resultsDir || path.join(
    __dirname,
    "../data/results"
  );
  const app = express();

  app.use(cors());
  app.use(express.json());
  app.use("/api/auth", authRoutes);
  app.get("/api/protected", authMiddleware, (req, res) => {
    res.json({ message: "You accessed a protected route!" });
  });
  app.use("/api", createResultsRouter({ resultsDir }));

  return app;
}

module.exports = { createApp };
