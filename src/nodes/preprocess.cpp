#include "nodes/preprocess.hpp"
#include "nodes/input.hpp"
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

PreprocessNode::PreprocessNode(const std::string &name, py::object py_func)
    : Node(NodeType::PREPROCESS, name, py_func) {}

void PreprocessNode::execute() {
  // Incomplete node (it should check for instance
  // with transform method as well)

  try {
    std::shared_ptr<InputNode> input = getInput();

    if (!input) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' requires a valid input");
    }

    py::object X = input->getX();
    py::object y = input->getY();

    if (X.is_none()) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' received None for X input");
    }

    py::object preprocessed_X;
    try {
      preprocessed_X = py_func(X);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in preprocessing: " +
                               std::string(e.what()));
    }

    if (should_create_new_data) {
      auto new_input = std::make_shared<InputNode>();
      new_input->setInputData(preprocessed_X, y);
      setOutput(new_input);
    } else {
      input->setInputData(preprocessed_X, y);
      setOutput(input);
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in preprocess node '" + name +
                             "': " + e.what());
  }
}

} // namespace mainera