#include <filesystem>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/constants.hpp"
#include "core/graph.hpp"
#include "nodes/input.hpp"
#include "nodes/preprocess.hpp"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace mainera {

PreprocessNode::PreprocessNode(const std::string &name, py::object py_func)
    : Node(NodeType::PREPROCESS, name, py_func) {}

std::tuple<py::object, py::object>
PreprocessNode::getInputData(std::shared_ptr<Node> input) {
  if (input->type == NodeType::INPUT) {
    auto input_node = std::dynamic_pointer_cast<InputNode>(input);
    if (!input_node) {
      throw std::runtime_error("Failed to cast to InputNode");
    }
    return {input_node->getX(), input_node->getY()};
  } else {
    std::shared_ptr<py::object> data_ptr = input->getData();
    if (!data_ptr || data_ptr->is_none()) {
      throw std::runtime_error("Input node data is None");
    }
    auto dict = data_ptr->cast<py::dict>();
    return {dict["X"], dict["y"]};
  }
}

void PreprocessNode::validateInputData(const py::object &X) {
  if (X.is_none()) {
    throw std::runtime_error("Preprocess node '" + name +
                             "' received None for X input");
  }
}

std::tuple<py::object, py::object> PreprocessNode::processData(py::object X,
                                                               py::object y) {

  if (py::hasattr(py_func, "fit")) {
    py::object transformer_instance =
        py::module::import("copy").attr("deepcopy")(py_func);

    py::module_ joblib = py::module_::import("joblib");
    py::object hash_obj =
        joblib.attr("hash")(py::make_tuple(transformer_instance, X));
    std::string hash_value = hash_obj.cast<std::string>();
    fs::path currentPath = fs::current_path();
    fs::path cacheDir = currentPath / CACHE_DIR;
    fs::create_directory(cacheDir);

    fs::path model_path = cacheDir / (hash_value + "model" + ".pkl");
    fs::path data_path = cacheDir / (hash_value + "data" + ".pkl");

    std::string modelPathStr = model_path.string();
    std::string dataPathStr = data_path.string();

    if (fs::exists(modelPathStr) && fs::exists(dataPathStr)) {
      transformer_instance = joblib.attr("load")(modelPathStr);
      X = joblib.attr("load")(dataPathStr);
      py_func = transformer_instance;
    } else {
      try {
        if (!getGraph()->getIsExecuted()) {
          transformer_instance.attr("fit")(X);
        }
        X = py::cast<py::array_t<double>>(
            transformer_instance.attr("transform")(X));
        py_func = transformer_instance;
        joblib.attr("dump")(transformer_instance, modelPathStr);
        joblib.attr("dump")(X, dataPathStr);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in model fitting: " +
                                 std::string(e.what()));
      }
    }
  } else {
    X = py_func(X);
  }

  return {X, y};
}

void PreprocessNode::storeResults(std::shared_ptr<Node> input,
                                  py::object processed_X,
                                  py::object processed_y) {
  if (should_create_new_data) {
    py::dict new_input;
    new_input["X"] = processed_X;
    new_input["y"] = processed_y;
    setData(std::make_shared<py::object>(new_input));
  } else {
    std::shared_ptr<py::object> data_ptr = input->getData();
    if (!data_ptr || data_ptr->is_none()) {
      throw std::runtime_error("Input node data is None");
    }
    auto dict = data_ptr->cast<py::dict>();
    dict["X"] = processed_X;
    dict["y"] = processed_y;
    setData(data_ptr);
  }
}

void PreprocessNode::execute() {
  try {
    auto input = getSourceNode();
    if (!input) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' requires a valid input");
    }

    auto [X, y] = getInputData(input);
    validateInputData(X);

    if (should_create_new_data) {
      X = X.attr("copy")();
      y = y.is_none() ? py::none() : y.attr("copy")();
      validateInputData(X);
    }

    auto [processed_X, processed_y] = processData(X, y);
    storeResults(input, processed_X, processed_y);

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in preprocess node '" + name +
                             "': " + e.what());
  }
}

void PreprocessNode::saveDataToDisc(const std::string &directory) {
  try {
    fs::path dirPath(directory);
    if (!fs::exists(dirPath)) {
      fs::create_directories(dirPath);
    }

    std::shared_ptr<py::object> data_ptr = getData();
    if (!data_ptr || data_ptr->is_none()) {
      throw std::runtime_error("No data available in preprocess node '" + name +
                               "'");
    }

    auto dict = data_ptr->cast<py::dict>();
    py::object X = dict["X"];
    py::object y = dict["y"];

    fs::path csvPath = dirPath / (name + ".csv");
    std::ofstream csvFile(csvPath.string());

    if (!csvFile.is_open()) {
      throw std::runtime_error("Failed to create CSV file: " +
                               csvPath.string());
    }

    py::array_t<double> X_array = py::cast<py::array_t<double>>(X);
    auto X_buf = X_array.request();

    size_t n_samples = X_buf.shape[0];
    size_t n_features = (X_buf.ndim > 1) ? X_buf.shape[1] : 1;

    for (size_t j = 0; j < n_features; ++j) {
      csvFile << "feature_" << j;
      if (j < n_features - 1)
        csvFile << ",";
    }
    if (!y.is_none()) {
      csvFile << ",label";
    }
    csvFile << "\n";

    double *X_ptr = static_cast<double *>(X_buf.ptr);

    py::array_t<double> y_array;
    double *y_ptr = nullptr;
    if (!y.is_none()) {
      y_array = py::cast<py::array_t<double>>(y);
      auto y_buf = y_array.request();
      y_ptr = static_cast<double *>(y_buf.ptr);
    }

    for (size_t i = 0; i < n_samples; ++i) {
      for (size_t j = 0; j < n_features; ++j) {
        size_t index = (X_buf.ndim > 1) ? (i * n_features + j) : i;
        csvFile << X_ptr[index];
        if (j < n_features - 1)
          csvFile << ",";
      }

      if (!y.is_none()) {
        csvFile << "," << y_ptr[i];
      }
      csvFile << "\n";
    }

    csvFile.close();

  } catch (const std::exception &e) {
    throw std::runtime_error("Error saving data to disc from node '" + name +
                             "': " + e.what());
  }
}

} // namespace mainera
