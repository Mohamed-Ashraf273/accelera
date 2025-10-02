#include <chrono>
#include <filesystem>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "core/graph.hpp"
#include "nodes/input.hpp"
#include "nodes/model.hpp"
#include "nodes/preprocess.hpp"

namespace fs = std::filesystem;
namespace py = pybind11;

namespace mainera {

ModelNode::ModelNode(const std::string &name, py::object py_func)
    : Node(NodeType::MODEL, name, py_func) {

  if (!py::hasattr(py_func, "fit")) {
    throw std::runtime_error("Model node '" + name +
                             "' must have a 'fit' method");
  }
}

std::tuple<py::object, py::object>
ModelNode::getInputData(std::shared_ptr<Node> input) {
  if (input->type == NodeType::INPUT) {
    auto input_node = std::dynamic_pointer_cast<InputNode>(input);
    if (!input_node) {
      throw std::runtime_error("Failed to cast to InputNode");
    }
    return {input_node->getX(), input_node->getY()};
  } else {
    auto data = input->getData();
    if (!data || data->is_none()) {
      throw std::runtime_error("PreprocessNode data is None");
    }
    auto dict = data->cast<py::dict>();
    return {dict["X"], dict["y"]};
  }
}

void ModelNode::validateInputData(const py::object &X, const py::object &y) {
  if (X.is_none()) {
    throw std::runtime_error("Model node '" + name +
                             "' received None inputs (X)");
  }

  if (!py::hasattr(X, "shape") || !(y.is_none() || py::hasattr(y, "shape"))) {
    throw std::runtime_error("Model node '" + name +
                             "' inputs must be array-like objects");
  }
}

py::object ModelNode::fitModel(py::object X, py::object y) {
  py::object model_instance =
      py::module::import("copy").attr("deepcopy")(py_func);

  py::module_ joblib = py::module_::import("joblib");
  py::object hash_obj =
      joblib.attr("hash")(py::make_tuple(model_instance, X, y));
  std::string hash_value = hash_obj.cast<std::string>();
  fs::path currentPath = fs::current_path();
  fs::path cacheDir = currentPath / ".mainera_cache";
  fs::create_directory(cacheDir);

  fs::path model_path = cacheDir / (hash_value + ".pkl");

  std::string modelPathStr = model_path.string();

  if (fs::exists(modelPathStr)) {
    model_instance = joblib.attr("load")(modelPathStr);
  } else {
    try {
      model_instance.attr("fit")(X, y);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in model fitting: " +
                               std::string(e.what()));
    }
    joblib.attr("dump")(model_instance, modelPathStr);
  }

  return model_instance;
}

void ModelNode::execute() {
  if (getGraph()->getIsExecuted()) {
    return;
  }

  try {
    auto input = getSourceNode();
    if (!input) {
      throw std::runtime_error("Model node '" + name +
                               "' requires a valid input node");
    }

    auto [X, y] = getInputData(input);
    validateInputData(X, y);

    auto fitted_model = std::make_shared<py::object>(fitModel(X, y));
    setData(fitted_model);

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("ModelNode: Python error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("ModelNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace mainera