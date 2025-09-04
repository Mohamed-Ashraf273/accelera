#include "nodes/preprocess.hpp"
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

PreprocessNode::PreprocessNode(const std::string &name, size_t numInputs,
                               size_t numOutputs, py::object py_func)
    : Node(NodeType::PREPROCESS, name, numInputs, numOutputs, py_func) {}

void PreprocessNode::execute() {
  // Incomplete node (it should check for instance
  // with transform method as well)

  try {
    py::list inputs = collectInputs();

    if (inputs.size() < 1) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' requires at least 1 input");
    }

    py::object X = inputs[0];
    py::object y;
    if (inputs.size() > 1) {
      y = inputs[1];
    } else {
      y = py::none();
    }

    if (X.is_none()) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' received None for X input");
    }

    // Apply preprocessing function to X
    py::object preprocessed_X;
    try {
      preprocessed_X = py_func(X);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in preprocessing: " +
                               std::string(e.what()));
    }

    // Set outputs directly to edges
    if (m_outputEdges.size() >= 2) {
      m_outputEdges[0]->setData(preprocessed_X);
      m_outputEdges[1]->setData(y);
    } else {
      setOutputs(preprocessed_X);
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in preprocess node '" + name +
                             "': " + e.what());
  }
}

} // namespace aistudio