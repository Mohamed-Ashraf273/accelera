#include "nodes/model.hpp"
#include "nodes/input.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>

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
    std::shared_ptr<InputNode> input = getInput();

    if (!input) {
      throw std::runtime_error("Model node '" + name +
                               "' requires a valid input");
    }

    py::object X = input->getX();
    py::object y = input->getY();

    if (X.is_none() || y.is_none()) {
      throw std::runtime_error("Model node '" + name +
                               "' received None inputs (X or y)");
    }

    if (!py::hasattr(X, "shape") || !py::hasattr(y, "shape")) {
      throw std::runtime_error("Model node '" + name +
                               "' inputs must be array-like objects");
    }

    py::object model_instance = py_func;

    if (should_create_new_data) {
      py::object X_copy = X.attr("copy")();
      py::object y_copy = y.attr("copy")();
      try {
        model_instance.attr("fit")(X_copy, y_copy);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in model fitting: " +
                                 std::string(e.what()));
      }
      auto new_input = std::make_shared<InputNode>();
      new_input->setInputData(X_copy, y_copy);
      new_input->setFittedModel(model_instance);
      setOutput(new_input);
    } else {
      try {
        model_instance.attr("fit")(X, y);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in model fitting: " +
                                 std::string(e.what()));
      }
      input->setFittedModel(model_instance);
      setOutput(input);
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