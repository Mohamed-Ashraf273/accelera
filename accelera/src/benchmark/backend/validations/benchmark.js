const isUrl = require("is-url");
const { spawn } = require("child_process");
const path = require("path");
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

const getJsonFromPythonOutput = (stdout, stderr) => {
  const lines = `${stdout}\n${stderr}`
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("{") && line.endsWith("}"));

  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      return JSON.parse(lines[i]);
    } catch (err) {
      continue;
    }
  }

  return null;
};

const run_python = (
  file_1,
  file_2,
  targetColumn,
  userId,
  which_python,
  sklearn_name,
  metric_paramters,
) => {
  return new Promise((resolve, reject) => {
    const projectRoot = path.resolve(__dirname, "../../../../..");
    const scriptPath = path.resolve(__dirname, `../scripts/${which_python}.py`);
    const pythonPath = process.env.PYTHONPATH
      ? `${projectRoot}:${process.env.PYTHONPATH}`
      : projectRoot;
    const pythonProcess = spawn(process.env.PYTHON_BIN || "python", [
      scriptPath,
      file_1,
      file_2,
      targetColumn,
      userId,
      sklearn_name,
      metric_paramters,
    ], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PYTHONPATH: pythonPath,
      },
    });
    let printedDataCorrectly = "";
    let printedError = "";
    let alreadyOccured = false;
    pythonProcess.stdout.on("data", (data) => {
      printedDataCorrectly += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      printedError += data.toString();
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

      const result = getJsonFromPythonOutput(printedDataCorrectly, printedError);
      if (!result) {
        return reject({
          message:
            printedError.trim() ||
            printedDataCorrectly.trim() ||
            "Invalid json",
          error: "No json response from python script",
        });
      }

      if (code !== 0) {
        return reject({
          message: result.message || printedError || "failed when run python",
          isValid: false,
        });
      }

      return resolve(result);
    });
  });
};
module.exports = {
  isUrlValidation,
  isValidProblemType,
  isGoogleDriveFileLink,
  run_python,
};
