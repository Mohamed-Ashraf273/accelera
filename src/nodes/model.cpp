#include "nodes/model.hpp"
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace aistudio {

ModelNode::ModelNode(const std::string &name, size_t numInputs,
                     size_t numOutputs, py::object py_func)
    : Node(NodeType::MODEL, name, numInputs, numOutputs, py_func) {

  // Validate that this is a model object with fit method
  if (!py::hasattr(py_func, "fit")) {
    throw std::runtime_error("Model node '" + name +
                             "' must have a 'fit' method");
  }
}

void ModelNode::execute() {
  try {
    py::list inputs = collectInputs();

    std::cout << "ModelNode " << name << " received " << inputs.size()
              << " inputs" << std::endl;

    if (inputs.size() < 2) {
      throw std::runtime_error("Model node '" + name +
                               "' requires at least 2 inputs (X, y)");
    }

    py::object X = inputs[0];
    py::object y = inputs[1];

    std::cout << "ModelNode " << name << " X type: " << py::str(X.get_type())
              << std::endl;
    std::cout << "ModelNode " << name << " y type: " << py::str(y.get_type())
              << std::endl;

    // Validate that inputs are arrays/tensors
    if (X.is_none() || y.is_none()) {
      throw std::runtime_error("Model node '" + name +
                               "' received None inputs");
    }

    // Check if inputs have array-like properties (shape, etc.)
    if (!py::hasattr(X, "shape") || !py::hasattr(y, "shape")) {
      throw std::runtime_error(
          "Model node '" + name +
          "' inputs must be array-like objects (X has shape: " +
          std::to_string(py::hasattr(X, "shape")) +
          ", y has shape: " + std::to_string(py::hasattr(y, "shape")) + ")");
    }

    py::object model_instance = py_func;
    model_instance.attr("fit")(X, y);

    setOutputs(model_instance);

  } catch (const py::error_already_set &e) {
    std::cerr << "ModelNode Python error: " << e.what() << std::endl;
    throw std::runtime_error("ModelNode: Python error during execution: " +
                             std::string(e.what()));

  } catch (const std::exception &e) {
    std::cerr << "ModelNode error: " << e.what() << std::endl;
    throw std::runtime_error("ModelNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace aistudio