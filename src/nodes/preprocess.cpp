#include "nodes/preprocess.hpp"
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

PreprocessNode::PreprocessNode(const std::string &name, size_t numInputs,
                               size_t numOutputs, py::object py_func)
    : Node(NodeType::PREPROCESS, name, numInputs, numOutputs, py_func) {}

void PreprocessNode::execute() {
  std::cout << "Executing Preprocess node: " << name << std::endl;

  try {
    py::list inputs = collectInputs();
    std::cout << "Collected " << inputs.size() << " inputs" << std::endl;

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

    // Simple validation - just check if X exists
    if (X.is_none()) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' received None for X input");
    }

    std::cout << "Applying preprocessing function..." << std::endl;

    // Apply preprocessing function to X
    py::object preprocessed_X;
    try {
      preprocessed_X = py_func(X);
      std::cout << "Preprocessing function applied successfully" << std::endl;
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in preprocessing: " +
                               std::string(e.what()));
    }

    // Set outputs
    if (m_outputEdges.size() >= 2) {
      // Output 0: preprocessed X
      m_outputEdges[0]->setData(preprocessed_X);
      std::cout << "Set preprocessed X to output 0" << std::endl;

      // Output 1: original y (pass through)
      m_outputEdges[1]->setData(y);
      std::cout << "Set y to output 1" << std::endl;
    } else if (m_outputEdges.size() == 1) {
      // Single output: preprocessed X
      m_outputEdges[0]->setData(preprocessed_X);
      std::cout << "Set preprocessed X to single output" << std::endl;
    }

    std::cout << "Preprocess node execution completed" << std::endl;

  } catch (const std::exception &e) {
    std::string error_msg =
        "Error in preprocess node '" + name + "': " + e.what();
    std::cout << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
}

} // namespace aistudio