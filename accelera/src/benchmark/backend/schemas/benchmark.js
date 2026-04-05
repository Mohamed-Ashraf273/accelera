const mongoose = require("mongoose");
const {
  isUrlValidation,
} = require("../validations/benchmark");
const benchmarkSchema = new mongoose.schema({
  title: {
    required: true,
    type: String,
    unique: true,
  },
  description: {
    required: true,
    type: String,
  },
  targetColumn: {
    required: true,
    type: String,
  },
  datasetLink: {
    required: true,
    type: String,
    validate: {
      validator: function (value) {
        return isUrlValidation(value);
      },
      message: (object) => `${object.value} is not a valid URL`,
    },
  },
  testSetWithoutPredictionsLink: {
    required: true,
    type: String,
    validate: {
      validator: function (value) {
        return isUrlValidation(value);
      },
      message: (object) => `${object.value} is not a valid URL`,
    },
  },
  predictedColumnLink: {
    required: true,
    type: String,
    validate: {
      validator: function (value) {
        return isUrlValidation(value);
      },
      message: (object) => `${object.value} is not a valid URL`,
    },
  },
  problemType: {
    required: true,
    type: String,
    enum: ["classification", "regression"],
  },
  evaluationMetric: {
    required: true,
    type: mongoose.Schema.Types.ObjectId,
    ref: "Metric",
  },
  creationDate: {
    type: Date,
    default: Date.now,
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
});
const Benchmark = mongoose.model("Benchmark", benchmarkSchema);
module.exports = Benchmark;

