#include <filesystem>
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

namespace accelera {

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

  if (py::hasattr(py_func["func"], "fit")) {
    py::object transformer_instance =
        py::module::import("copy").attr("deepcopy")(py_func["func"]);

    py::object execute_fit =
        py::module::import("copy").attr("deepcopy")(py_func["execute_fit"]);

    py::module_ joblib = py::module_::import("joblib");
    py::object hash_obj =
        joblib.attr("hash")(py::make_tuple(transformer_instance, X));
    std::string hash_value = hash_obj.cast<std::string>();
    fs::path currentPath = fs::current_path();
    fs::path cacheDir = currentPath / CACHE_DIR;
    fs::create_directory(cacheDir);

    fs::path model_path = cacheDir / (hash_value + "model" + ".pkl");

    std::string modelPathStr = model_path.string();

    if (fs::exists(modelPathStr)) {
      transformer_instance = joblib.attr("load")(modelPathStr);
      X = py::cast<py::array_t<double>>(
          transformer_instance.attr("transform")(X));
      py_func["func"] = transformer_instance;
    } else {
      try {
        if (!getGraph()->getIsExecuted()) {
          transformer_instance = execute_fit(transformer_instance, X, y);
        }
        X = py::cast<py::array_t<double>>(
            transformer_instance.attr("transform")(X));
        py_func["func"] = transformer_instance;
        joblib.attr("dump")(transformer_instance, modelPathStr);
        // Don't cache transformed data - it uses too much memory
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in model fitting: " +
                                 std::string(e.what()));
      }
    }
  } else {
    X = py_func["func"](X);
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

    // Only copy data if it will be modified in-place by the transformer
    // Most sklearn transformers return new arrays, so no copy needed
    if (should_create_new_data && py::hasattr(py_func["func"], "inplace") &&
        py::cast<bool>(py_func["func"].attr("inplace"))) {
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

} // namespace accelera
