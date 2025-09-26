#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nodes/model.hpp"

#include "nodes/input.hpp"
#include "nodes/preprocess.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

namespace py = pybind11;

namespace mainera {

ModelNode::ModelNode(const std::string &name, py::object py_func)
    : Node(NodeType::MODEL, name, py_func) {
  // Validate that this is a model object with fit method
  if (!py::hasattr(py_func, "fit")) {
    throw std::runtime_error("Model node '" + name +
                             "' must have a 'fit' method");
  }
}

void ModelNode::execute() {
  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input && input->type != NodeType::PREPROCESS) {
      throw std::runtime_error("Model node '" + name +
                               "' requires a valid preprocess node");
    }

    std::shared_ptr<PreprocessNode> input_node =
        std::dynamic_pointer_cast<PreprocessNode>(input);
    if (!input_node) {
      throw std::runtime_error("Failed to cast to PreprocessNode");
    }

    std::shared_ptr<InputNode> data = input_node->getData();

    py::object X = data->getX();
    py::object y = data->getY();

    if (X.is_none() || y.is_none()) {
      throw std::runtime_error("Model node '" + name +
                               "' received None inputs (X or y)");
    }

    if (!py::hasattr(X, "shape") || !py::hasattr(y, "shape")) {
      throw std::runtime_error("Model node '" + name +
                               "' inputs must be array-like objects");
    }

    py::object model_instance =
        py::module::import("copy").attr("deepcopy")(py_func);

    py::module_ joblib = py::module_::import("joblib");
    py::module_ np = py::module_::import("numpy");
    namespace fs = std::filesystem;
    py::object hash_obj =
        joblib.attr("hash")(py::make_tuple(model_instance, X, y));
    std::string hash_value = hash_obj.cast<std::string>();
    std::string currentPath = fs::current_path().string();
    fs::create_directory(currentPath + "\\cache");
    std::string model_path = currentPath + "\\cache\\" + hash_value + ".pkl";

    if (fs::exists(model_path)) {
      py::object fitted = joblib.attr("load")(model_path);
      setData(fitted);
    } else {
      try {
        model_instance.attr("fit")(X, y);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in model fitting: " +
                                 std::string(e.what()));
      }
      setData(model_instance);
      joblib.attr("dump")(model_instance, model_path);
    }
  } catch (const py::error_already_set &e) {
    throw std::runtime_error("ModelNode: Python error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("ModelNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace mainera