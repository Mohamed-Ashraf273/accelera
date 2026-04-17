const express = require("express");
const router = express.Router();
const Benchmark = require("../schemas/benchmark");
const Submission = require("../schemas/submissions");
const User = require("../schemas/user");
const {
  isUrlValidation,
  isGoogleDriveFileLink,
  run_python,
} = require("../validations/benchmark");
router.get("/benchmark/:id", async (req, res) => {
  try {
    const benchmarkID = req.params.id;
    const benchmark = await Benchmark.findById(benchmarkID).populate(
      "evaluationMetric",
      "whichBetter",
    );
    if (!benchmark) {
      return res.status(404).json({
        message: `There is no benchmark with this id: ${benchmarkID}`,
      });
    }
    const typeSort = benchmark.evaluationMetric.whichBetter;

    const submissions = await Submission.find({ benchmarkId: benchmarkID })
      .select("score submittedBy submissionDate")
      .populate("submittedBy", "name email")
      .sort(typeSort === "higher" ? "-score" : "score");

    return res.status(200).json(submissions);
  } catch (err) {
    console.error("Error while fetching submissions for this benchmark:", err);
    return res
      .status(500)
      .json({ message: "Error while fetching submissions for this benchmark" });
  }
});
router.get("/user/:id", async (req, res) => {
  try {
    const userId = req.params.id;
    const submissions = await Submission.find({ submittedBy: userId })
      .select("score submissionDate benchmark")
      .populate("benchmarkId", "title");
    return res.status(200).json(submissions);
  } catch (err) {
    console.error("Error while fetching submissions:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching submissions" });
  }
});

router.get("/:id", async (req, res) => {
  try {
    const submissionId = req.params.id;
    const submission = await Submission.findById(submissionId)
      .populate("submittedBy", "name email")
      .populate("benchmarkId", "title");
    if (!submission) {
      return res.status(404).json({
        message: `There is no submission with this id: ${submissionId}`,
      });
    }
    return res.status(200).json(submission);
  } catch (err) {
    console.error("Error while fetching submission:", err);
    return res
      .status(500)
      .json({ message: "There is an error while fetching submission" });
  }
});

router.post("/:benchmarkId", async (req, res) => {
  try {
    const benchmarkId = req.params.benchmarkId;
    let { submittedBy, repoLink, predictedColumnLink } = req.body;
    const existSubmission = await Submission.find({
      submittedBy: submittedBy,
      benchmarkId: benchmarkId,
    });
    if (!!existSubmission) {
      return res.status(400).json({
        message: `This submission is already exist`,
      });
    }

    const benchmark = await Benchmark.findById(benchmarkId)
      .select(
        "targetColumn predictedColumnLink metricPramaters evaluationMetric",
      )
      .populate("evaluationMetric", "sklearnMetricName");
    if (!benchmark) {
      return res.status(400).json({
        message: `This id ${benchmarkId} is not exist`,
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
    if (!isUrlValidation(repoLink)) {
      return res.status(400).json({
        message: `This link ${repoLink}  is not a valid link`,
      });
    }

    const results = await run_python(
      benchmark.predictedColumnLink,
      predictedColumnLink,
      benchmark.targetColumn,
      submittedBy,
      "get_score",
      benchmark.evaluationMetric.sklearnMetricName,
      JSON.stringify(Object.fromEntries(benchmark.metricPramaters)),
    );
    if (results.isValid === false) {
      return res.status(400).json({
        message: results.message,
      });
    }
    const score = results.score;
    const newSubmission = await Submission.create({
      benchmarkId,
      submittedBy,
      repoLink,
      predictedColumnLink,
      score,
    });
    console.log(newSubmission);
    res.status(201).json(newSubmission);
  } catch (err) {
    console.error("Error while creating Submission:", err);
    res
      .status(500)
      .json({ message: `There is an error while creating Submission ` });
  }
});
router.patch("/:id", async (req, res) => {
  try {
    const submissionId = req.params.id;
    const { predictedColumnLink } = req.body;
    if (
      !isUrlValidation(predictedColumnLink) ||
      !isGoogleDriveFileLink(predictedColumnLink)
    ) {
      return res.status(400).json({
        message: `This link ${predictedColumnLink}  is not a valid google drive file link`,
      });
    }
    const submission = await Submission.findById(submissionId).populate({
      path: "benchmarkId",
      populate: {
        path: "evaluationMetric",
        select: "sklearnMetricName",
      },
    });
    if (!submission) {
      return res.status(400).json({
        message: `This id ${submissionId} is not exist`,
      });
    }

    const results = await run_python(
      submission.benchmarkId.predictedColumnLink,
      predictedColumnLink,
      submission.benchmarkId.targetColumn,
      submission.submittedBy,
      "get_score",
      submission.benchmarkId.evaluationMetric.sklearnMetricName,
      JSON.stringify(
        Object.fromEntries(submission.benchmarkId.metricPramaters),
      ),
    );
    if (results.isValid === false) {
      return res.status(400).json({
        message: results.message,
      });
    }
    submission.score = results.score;
    submission.predictedColumnLink = predictedColumnLink;
    await submission.save();
    return res.json(submission);
  } catch (error) {
    console.error("Error while updating submission:", error);
    return res
      .status(500)
      .json({ message: "There is an error while updating submission" });
  }
});
router.delete("/:id", async (req, res) => {
  try {
    const submissionId = req.params.id;

    const submission = await Submission.findByIdAndDelete(submissionId);
    if (!submission) {
      return res.status(404).json({
        message: `There is no submission for this id: ${submissionId}`,
      });
    }
    return res.status(200).json({ message: "submission successfully deleted" });
  } catch (error) {
    console.error("Error while deleting submission:", error);
    return res
      .status(500)
      .json({ message: "There is an error while deleting submission" });
  }
});

module.exports = router;
