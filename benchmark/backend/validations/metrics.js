const Metric = require("../schemas/metrics")

const isExistingMetric = async (name, problemType, sklearnMetricName) => {
    const metric = await Metric.findOne({ name, problemType, sklearnMetricName })
    if (metric)
        return true
    return false
}
const isValidWhichBetter = (whichBetter) => {
    const allowedValues = ["higher", "lower"];
    return allowedValues.includes(whichBetter);
}   
module.exports = { isExistingMetric, isValidWhichBetter }