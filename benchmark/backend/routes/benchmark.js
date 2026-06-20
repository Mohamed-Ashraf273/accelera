const express = require("express");
const router = express.Router();
const Benchmark = require("../schemas/benchmark");
const Submission = require("../schemas/submissions");
const auth = require("../middleware/auth");
const {
  isUrlValidation,
  isValidProblemType,
  isGoogleDriveFileLink,
  run_python,
} = require("../validations/benchmark");
router.get("/", async (req, res) => {
  try {
    const benchmarks = await Benchmark.find()
      .select(
        "title problemType targetColumn testSetWithoutPredictionsLink datasetLink description creationDate metricParamaters evaluationMetric createdBy",
      )
      .populate("createdBy", "name email")
      .populate("evaluationMetric", "name");
    return res.status(200).json(benchmarks);
  } catch (err) {
    console.error("Error while fetching Benchmarks:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching Benchmarks" });
  }
});
router.get("/user/:id", async (req, res) => {
  try {
    const userId = req.params.id;
    const benchmarks = await Benchmark.find({ createdBy: userId })
      .select(
        "title problemType targetColumn testSetWithoutPredictionsLink datasetLink description creationDate metricParamaters evaluationMetric createdBy",
      )
      .populate("evaluationMetric", "name")
      .populate("createdBy", "name email");
    return res.status(200).json(benchmarks);
  } catch (err) {
    console.error("Error while fetching Benchmarks:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching Benchmarks" });
  }
});
router.get("/problem-type/:problemType", async (req, res) => {
  try {
    let problemType = req.params.problemType;
    problemType = problemType.toLowerCase();
    if (!isValidProblemType(problemType)) {
      return res.status(400).json({
        message: `Invalid problem type must be classification or regression`,
      });
    }
    const benchmarks = await Benchmark.find({
      problemType: problemType,
    })
      .select(
        "title problemType targetColumn testSetWithoutPredictionsLink datasetLink description creationDate metricParamaters evaluationMetric createdBy",
      )
      .populate("createdBy", "name email")
      .populate("evaluationMetric", "name");
    return res.status(200).json(benchmarks);
  } catch (err) {
    console.error("Error while fetching Benchmarks:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching Benchmarks" });
  }
});
router.get("/:id", async (req, res) => {
  try {
    const benchmarkId = req.params.id;
    const benchmark = await Benchmark.findById(benchmarkId)
      .populate("createdBy", "name email")
      .populate("evaluationMetric", "name whichBetter");
    return res.status(200).json(benchmark);
  } catch (err) {
    console.error("Error while fetching Benchmarks:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching Benchmarks" });
  }
});

router.post("/", auth, async (req, res) => {
  try {
    let {
      title,
      description,
      targetColumn,
      datasetLink,
      testSetWithoutPredictionsLink,
      predictedColumnLink,
      problemType,
      evaluationMetric,
      metricParamaters,
    } = req.body;
    const createdBy = req.user._id;

    problemType = problemType.toLowerCase();
    if (!isValidProblemType(problemType)) {
      return res.status(400).json({
        message: `Invalid problem type must be classification or regression`,
      });
    }
    const benchmark = await Benchmark.findOne({ title: title });
    if (!!benchmark) {
      return res.status(400).json({
        message: `This title ${title} is already exist`,
      });
    }
    if (
      !isUrlValidation(predictedColumnLink) ||
      !isGoogleDriveFileLink(predictedColumnLink)
    ) {
      return res.status(400).json({
        message: `This link ${predictedColumnLink}  is not a valid google drive file link`,
      });
    }
    if (!isUrlValidation(datasetLink)) {
      return res.status(400).json({
        message: `This link ${datasetLink}  is not a valid link`,
      });
    }
    if (
      !isUrlValidation(testSetWithoutPredictionsLink) ||
      !isGoogleDriveFileLink(testSetWithoutPredictionsLink)
    ) {
      return res.status(400).json({
        message: `This link ${testSetWithoutPredictionsLink}  is not a valid google drive file link`,
      });
    }
    if (
      !isUrlValidation(predictedColumnLink) ||
      !isGoogleDriveFileLink(predictedColumnLink)
    ) {
      return res.status(400).json({
        message: `This link ${predictedColumnLink}  is not a valid google drive file link`,
      });
    }
    const targetResults = await run_python(
      testSetWithoutPredictionsLink,
      predictedColumnLink,
      targetColumn,
      createdBy,
      "validate_user_links",
      undefined,
      undefined,
    );
    if (targetResults.isValid === false) {
      return res.status(400).json({
        message: targetResults.message,
      });
    }
    const newBenchmark = await Benchmark.create({
      title,
      description,
      targetColumn,
      datasetLink,
      testSetWithoutPredictionsLink,
      predictedColumnLink,
      problemType,
      evaluationMetric,
      metricParamaters,
      createdBy,
    });
    return res.status(201).json(newBenchmark);
  } catch (err) {
    console.error("Error while creating Benchmarks:", err);
    if (err.message) {
      const statusCode = err.isValid === false ? 400 : 500;
      return res.status(statusCode).json({ message: err.message });
    }
    return res
      .status(500)
      .json({ message: `there is an error while creating Benchmarks ` });
  }
});

router.delete("/:id", auth, async (req, res) => {
  try {
    const benchmarkId = req.params.id;
    const benchmarkInfo = await Benchmark.findById(benchmarkId);
    if (!benchmarkInfo) {
      return res
        .status(404)
        .json({ message: `no benchmark for this id: ${benchmarkId}` });
    }
    if (
      req.user.role !== "admin" &&
      benchmarkInfo.createdBy.toString() !== req.user._id.toString()
    ) {
      return res.status(403).json({
        message: "You are not allowed to delete this benchmark",
      });
    }
    const submissionsCount = await Submission.countDocuments({
      benchmarkId: benchmarkId,
    });

    if (submissionsCount > 0) {
      return res.status(400).json({
        message:
          "Cannot delete this benchmark because there are submissions in this benchmark",
      });
    }

    await Benchmark.findByIdAndDelete(benchmarkId);
    return res.status(200).json({ message: "benchmark successfully deleted" });
  } catch (error) {
    console.error("Error while deleting benchmark:", error);
    return res
      .status(500)
      .json({ message: "There is an error while deleting benchmark" });
  }
});

module.exports = router;
