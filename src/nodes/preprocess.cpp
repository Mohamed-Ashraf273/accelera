#include "nodes/preprocess.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

PreprocessNode::PreprocessNode(const std::string &name, size_t numInputs,
                               size_t numOutputs, py::object py_func)
    : Node(NodeType::PREPROCESS, name, numInputs, numOutputs, py_func) {}

void PreprocessNode::execute() {
  // Implement the preprocessing logic here
  // This should use the py_func for data preprocessing
}

} // namespace aistudio