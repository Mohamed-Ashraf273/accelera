const isUrl = require("is-url");

const isUrlValidation = (value) => {
  return isUrl(value);
};
const isValidProblemType = (problemType) => {
  const allowed_problems = ["classification", "regression"];
  if (!allowed_problems.includes(problemType)) {
    return false;
  }
  return true;
};

const isGoogleDriveFileLink = (link) => {
  return /drive\.google\.com\/file\/d\/.+/.test(link);
};
module.exports = {
  isUrlValidation,
  isValidProblemType,

  isGoogleDriveFileLink,
};
