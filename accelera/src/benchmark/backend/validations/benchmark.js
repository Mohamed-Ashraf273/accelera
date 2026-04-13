const isUrl = require("is-url");
const { spawn } = require("child_process");
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
const ignoreWrongPrints = (printed) => {
  if (
    printed.includes("%|") ||
    printed.includes("Downloading") ||
    printed.includes("Cache") ||
    printed.includes("Downloaded")
  ) {
    return true;
  }
  return false;
};

const isValidUserLink = (
  test_link,
  target_link,
  targetColumn,
  userId
) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python", [
      "scripts/validate_user_links.py",
      test_link,
      target_link,
      targetColumn,
      userId,
    ]);
  
    let printedDataCorrectly = "";
    let printedError = "";
    let alreadyOccured = false;

    pythonProcess.stdout.on("data", (data) => {
      const printed = data.toString();
      if (ignoreWrongPrints(printed)) return;
      printedDataCorrectly += printed;
    });

    pythonProcess.stderr.on("data", (data) => {
      const printed = data.toString();
      if (ignoreWrongPrints(printed)) return;
      printedError += printed;
    });

    pythonProcess.on("error", (err) => {
      if (!alreadyOccured) {
        alreadyOccured = true;
        reject(`error when try to run python file ${err.message}`);
      }
    });

    pythonProcess.on("close", (code) => {
      if (alreadyOccured) return;
      alreadyOccured = true;

      try {
        const result = JSON.parse(printedDataCorrectly);
        if (code !== 0) {
          return reject({
            message: result.message || printedError || "failed when run python",
            isValid: false,
          });
        }

        return resolve({
          message: result.message,
          isValid: result.isValid,
        });
       
      } catch (err) {
        return reject({
          message: "Invalid json",
          error: err.message,
        });
      }
    });
  });
};
module.exports = {
  isUrlValidation,
  isValidProblemType,
  isGoogleDriveFileLink,
  isValidUserLink,
};
