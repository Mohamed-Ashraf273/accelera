const express = require("express");
const router = express.Router();
const Benchmark = require("../schemas/benchmark");
const Submission = require("../schemas/submissions");
const User = require("../schemas/user");
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

    const submissions = await Submission.find({ benchmark: benchmarkID })
      .select("score submittedBy submissionDate")
      .populate("submittedBy", "name email");
    if (typeSort === "higher") {
      submissions.sort((a, b) => b.score - a.score);
    } else {
      submissions.sort((a, b) => a.score - b.score);
    }
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
      .populate("benchmark", "title");
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
      .populate("benchmark", "title");
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

// router.post("/", async (req, res) => {
//   try {
//     let {
//       title,
//       description,
//       targetColumn,
//       datasetLink,
//       testSetWithoutPredictionsLink,
//       predictedColumnLink,
//       problemType,
//       evaluationMetric,
//       createdBy,
//     } = req.body;
//     problemType = problemType.toLowerCase();
//     if (!isValidProblemType(problemType)) {
//       return res.status(400).json({
//         message: `Invalid problem type must be classification or regression`,
//       });
//     }
//     const benchmark = await Benchmark.findOne({ title: title });
//     if (!!benchmark) {
//       return res.status(400).json({
//         message: `This title ${title} is already exist`,
//       });
//     }
//     if (
//       !isUrlValidation(predictedColumnLink) ||
//       !isGoogleDriveFileLink(predictedColumnLink)
//     ) {
//       return res.status(400).json({
//         message: `This link ${predictedColumnLink}  is not a valid google drive file link`,
//       });
//     }
//     if (!isUrlValidation(datasetLink)) {
//       return res.status(400).json({
//         message: `This link ${datasetLink}  is not a valid link`,
//       });
//     }
//     if (
//       !isUrlValidation(testSetWithoutPredictionsLink) ||
//       !isGoogleDriveFileLink(testSetWithoutPredictionsLink)
//     ) {
//       return res.status(400).json({
//         message: `This link ${testSetWithoutPredictionsLink}  is not a valid google drive file link`,
//       });
//     }
//     if (
//       !isUrlValidation(predictedColumnLink) ||
//       !isGoogleDriveFileLink(predictedColumnLink)
//     ) {
//       return res.status(400).json({
//         message: `This link ${predictedColumnLink}  is not a valid google drive file link`,
//       });
//     }
//     const targetResults = await isValidUserLink(
//       testSetWithoutPredictionsLink,
//       predictedColumnLink,
//       targetColumn,
//       createdBy,
//     );
//     if (targetResults.isValid === false) {
//       return res.status(400).json({
//         message: targetResults.message,
//       });
//     }
//     const newBenchmark = await Benchmark.create({
//       title,
//       description,
//       targetColumn,
//       datasetLink,
//       testSetWithoutPredictionsLink,
//       predictedColumnLink,
//       problemType,
//       evaluationMetric,
//       createdBy,
//     });
//     res.status(201).json(newBenchmark);
//   } catch (err) {
//     console.error("Error while creating Benchmarks:", err);
//     res
//       .status(500)
//       .json({ message: `There is an error while creating Benchmarks ` });
//   }
// });

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
