const mongoose = require("mongoose");
const metricSChema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  sklearnMetricName: {
    type: String,
    required: true,
  },
  neededParameters: {
    type: Map,
    of: { type: [String], default: [] },
    default: {},
  },
  problemType: {
    type: String,
    required: true,
    enum: ["classification", "regression"],
  },
  whichBetter: {
    required: true,
    type: String,
    enum: ["higher", "lower"],
  },
});
const Metric = mongoose.model("Metric", metricSChema);
module.exports = Metric;
