const mongoose = require("mongoose");
require("dotenv").config();

const { createApp } = require("./app");

const PORT = Number(process.env.PORT) || 5000;

async function startServer() {
  if (process.env.MONGO_URI) {
    try {
      await mongoose.connect(process.env.MONGO_URI);
      console.log("MongoDB connected");
    } catch (error) {
      console.error("MongoDB connection failed");
    }
  } else {
    console.log("MongoDB connection skipped because MONGO_URI is not set");
  }

  return createApp().listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

if (require.main === module) {
  startServer();
}

module.exports = { startServer };
