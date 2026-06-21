const express = require("express");
const connectMongo = require("./config/create_db");
const getJson = require("./config/get_json");
const metricsRoutes = require("./routes/metrics");
const bechmarkRoutes = require("./routes/benchmark");
const UserRoutes = require("./routes/user");
const SubmissionRoutes=require("./routes/submissions")
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());
async function runServer() {
  try {
    const jsonFile =await  getJson()
    connectMongo(jsonFile.Mongos_url);
    app.use("/metrics", metricsRoutes);
    app.use("/benchmark", bechmarkRoutes);
    app.use("/user", UserRoutes);
    app.use("/submission", SubmissionRoutes);
    const port = jsonFile.DB_PORT ;
    app.listen(port, () => {
      console.log(`The server is running on port ${port}`);
    });
  } catch (err) {
    console.error("Failed to start server:", err);
  }
}
runServer()



