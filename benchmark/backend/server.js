const express = require("express");
const connectMongo = require("./config/create_db");
const metricsRoutes = require("./routes/metrics");
const bechmarkRoutes = require("./routes/benchmark");
const UserRoutes = require("./routes/user");
const SubmissionRoutes=require("./routes/submissions")
const dotenv = require("dotenv");
const cors = require("cors");
dotenv.config({ path: ".env", quiet: true });
dotenv.config({ path: "env", quiet: true });
const app = express();
app.use(cors());
app.use(express.json());
connectMongo();
app.use("/metrics", metricsRoutes);
app.use("/benchmark", bechmarkRoutes);
app.use("/user", UserRoutes);
app.use("/submission", SubmissionRoutes);
const port = process.env.DB_PORT || 3000;
app.listen(port, () => {
  console.log(`The server is running on port ${port}`);
});


