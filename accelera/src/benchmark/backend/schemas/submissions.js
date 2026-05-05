const mongoose = require("mongoose");
const { isUrlValidation } = require("../validations/benchmark");
const submissionSchema = new mongoose.Schema({
  benchmarkId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Benchmark",
    required: true,
  },
  submittedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  repoLink: {
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
  submissionDate: {
    type: Date,
    default: Date.now,
  },
  score: {
    type: Number,
    required: true,
  },
});
submissionSchema.index({ benchmarkId: 1, score: -1 });

const Submission = mongoose.model("Submission", submissionSchema);
module.exports = Submission;
