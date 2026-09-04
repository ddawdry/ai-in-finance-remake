const { createApp } = require("./app");

const PORT = Number(process.env.PORT) || 5000;
const HOST = process.env.HOST || "127.0.0.1";

function startServer() {
  return createApp().listen(PORT, HOST, () => {
    console.log(`Server running at http://${HOST}:${PORT}`);
  });
}

if (require.main === module) {
  startServer();
}

module.exports = { startServer };
