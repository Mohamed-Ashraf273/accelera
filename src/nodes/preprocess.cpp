#include "nodes/preprocess.hpp"
#include "nodes/input.hpp"
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

    if (should_create_new_data) {
      py::object X_copy = X.attr("copy")();
      py::object y_copy = y.attr("copy")();
      if (X_copy.is_none()) {
        throw std::runtime_error("Preprocess node '" + name +
                                 "' received None for X input");
      }

      try {
        X_copy = py_func(X_copy);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in preprocessing: " +
                                 std::string(e.what()));
      }
      auto new_input = std::make_shared<InputNode>();
      new_input->setInputData(X_copy, y_copy);
      setOutput(new_input);
    } else {

      if (X.is_none()) {
        throw std::runtime_error("Preprocess node '" + name +
                                 "' received None for X input");
      }

      try {
        X = py_func(X);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in preprocessing: " +
                                 std::string(e.what()));
      }
      input->setInputData(X, y);
      setOutput(input);
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in preprocess node '" + name +
                             "': " + e.what());
  }
}

} // namespace mainera