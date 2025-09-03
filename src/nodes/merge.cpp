#include "nodes/merge.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

MergeNode::MergeNode(const std::string &name, size_t numInputs,
                     size_t numOutputs, py::object py_func)
    : Node(NodeType::MERGE, name, numInputs, numOutputs, py_func) {}

void MergeNode::execute() {
  try {
    py::list inputs = collectInputs();

    if (inputs.size() == 0) {
      throw std::runtime_error("Merge node '" + name + "' received no inputs");
    }

    // If only one input, just pass it through
    if (inputs.size() == 1) {
      m_result = inputs[0];
      if (!getOutputEdges().empty()) {
        setOutputs(inputs[0]);
      }
      return;
    }

    // Apply the merge function to the inputs
    py::object result;
    try {
      result = py_func(inputs);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in merge function: " +
                               std::string(e.what()));
    }

    // Store the result for later retrieval
    m_result = result;

    // If we have output edges, also set the outputs
    if (!getOutputEdges().empty()) {
      setOutputs(result);
    }

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("MergeNode: Error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("MergeNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace aistudio
