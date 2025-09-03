#include "nodes/merge.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

MergeNode::MergeNode(const std::string &name, size_t numInputs,
                     size_t numOutputs, py::object py_func)
    : Node(NodeType::MERGE, name, numInputs, numOutputs, py_func) {}

void MergeNode::execute() {
  // Implement the merge logic here
  // This should use the py_func for merging
}

} // namespace aistudio
