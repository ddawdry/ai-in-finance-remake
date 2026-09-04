const express = require("express");
const cors = require("cors");
const path = require("path");

const { createResultsRouter } = require("./routes/results");

function createApp(options = {}) {
  const resultsDir = options.resultsDir || path.join(
    __dirname,
    "../data/results"
  );
  const app = express();

  app.use(cors());
  app.use("/api", createResultsRouter({ resultsDir }));

  return app;
}

module.exports = { createApp };
