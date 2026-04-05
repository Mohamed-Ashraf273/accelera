const express = require("express")
const router = express.Router()
const Metric = require("../schemas/metrics")
const {
  isExistingMetric,
  isValidWhichBetter,
} = require("../validations/metrics")
router.get("/problem-type/:problemType", async (req, res) => {
  try {
    let problemType = req.params.problemType
    problemType = problemType.toLowerCase()
    const metrics = await Metric.find({ problemType: problemType })
    if (metrics.length === 0) {
      return res.status(404).json({
        message: `There is no metric for this problem type: ${problemType}`,
      })
    }
    res.status(200).json(metrics)
  } catch (error) {
    console.error("Error while fetching metrics:", error)
    res
      .status(500)
      .json({ message: "There is an error while fetching metrics" })
  }
})

router.get("/:id", async (req, res) => {
  try {
    const metricId = req.params.id
    const metric = await Metric.findById(metricId)
    if (!metric) {
      return res
        .status(404)
        .json({ message: `There is no metric for this id: ${metricId}` })
    }
    res.status(200).json(metric)
  } catch (error) {
    console.error("Error while fetching metric:", error)
    res
      .status(500)
      .json({ message: "There is an error while fetching metric" })
  }
})

router.post("/", async (req, res) => {
  try {
    let {
      name,
      sklearnMetricName,
      problemType,
      whichBetter,
      neededParameters,
    } = req.body

    if (isExistingMetric(name, problemType, sklearnMetricName) == true) {
      return res.status(400).json({
        message: `There is already a metric with name ${name} and sklearn metric name ${sklearnMetricName} with problem type ${problemType}`,
      })
    }
    whichBetter = whichBetter.toLowerCase()
    problemType = problemType.toLowerCase()
    if (!isValidWhichBetter(whichBetter)) {
      return res.status(400).json({
        message: `wrong which better value allowed values are "higher" or "lower" in any case`,
      })
    }

    const metric = new Metric({
      name,
      sklearnMetricName,
      neededParameters,
      problemType,
      whichBetter,
    })
    await metric.save()
    res.status(201).json(metric)
  } catch (error) {
    console.error("Error while creating metric:", error)
    res
      .status(500)
      .json({ message: "There is an error while creating metric" })
  }
})
router.put("/:id", async (req, res) => {
  try {
    const metricId = req.params.id
    let {
      name,
      sklearnMetricName,
      problemType,
      whichBetter,
      neededParameters,
    } = req.body
    problemType = problemType.toLowerCase()
    whichBetter = whichBetter.toLowerCase()
    const metric = await Metric.findByIdAndUpdate(
      metricId,
      {
        name,
        sklearnMetricName,
        neededParameters,
        problemType,
        whichBetter,
      },
      { new: true },
    )
    if (!metric) {
      return res
        .status(404)
        .json({ message: `There is no metric for this id: ${metricId}` })
    }
    res.status(200).json(metric)
  } catch (error) {
    console.error("Error while updating metric:", error)
    res
      .status(500)
      .json({ message: "There is an error while updating metric" })
  }
})
router.delete("/:id", async (req, res) => {
  try {
    const metricId = req.params.id
    const metric = await Metric.findByIdAndDelete(metricId)
    if (!metric) {
      return res
        .status(404)
        .json({ message: `There is no metric for this id: ${metricId}` })
    }
    res.status(200).json({ message: "Metric successfully deleted" })
  } catch (error) {
    console.error("Error while deleting metric:", error)
    res
      .status(500)
      .json({ message: "There is an error while deleting metric" })
  }
})

module.exports = router

