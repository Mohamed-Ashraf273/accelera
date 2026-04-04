const isUrl = require('is-url');
const evaluationMatrixValidation = (value, problemType) => {
    if (problemType === 'classification') {
        const allowedMetrics = ['accuracy', 'precision', 'recall', 'f1-score', 'area_under_curve'];
        return allowedMetrics.includes(value);
    }
    if (problemType === 'regression') {
        const allowedMetrics = ['mean_squared_error', 'mean_absolute_error', 'r2_score'];
        return allowedMetrics.includes(value);
    }
    return false;
}
const isUrlValidation = (value) => {
    return isUrl(value);
}
module.exports = { evaluationMatrixValidation, isUrlValidation }