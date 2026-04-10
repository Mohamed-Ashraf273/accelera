const express = require("express");
const connectMongo = require("./config/create_db");
const metricsRoutes = require("./routes/metrics");
const bechmarkRoutes = require("./routes/benchmark");
const UserRoutes = require("./routes/user");
const dotenv = require("dotenv");
const cors = require("cors");
dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());
connectMongo();
app.use("/metrics", metricsRoutes);
app.use("/benchmark", bechmarkRoutes);
app.use("/user", UserRoutes);
app.listen(process.env.PORT, () => {
  console.log(`The server is running on port ${process.env.PORT}`);
});


