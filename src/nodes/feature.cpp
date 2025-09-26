#include <pybind11/pybind11.h>

#include "nodes/feature.hpp"

namespace py = pybind11;

namespace mainera {

FeatureNode::FeatureNode(const std::string &name, py::object py_func)
    : Node(NodeType::FEATURE, name, py_func) {}

void FeatureNode::execute() {
  // Implement the feature extraction logic here
  // This should use the py_func for feature extraction
}

} // namespace mainera