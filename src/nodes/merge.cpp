#include <pybind11/pybind11.h>

#include "nodes/input.hpp"
#include "nodes/merge.hpp"

namespace py = pybind11;

namespace mainera {

MergeNode::MergeNode(const std::string &name, py::object py_func)
    : Node(NodeType::MERGE, name, py_func) {}

void MergeNode::execute() {
  throw std::runtime_error("Merge node '" + name + "' is not implemented yet");
}

} // namespace mainera
