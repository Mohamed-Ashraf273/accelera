#include "nodes/metric.hpp"
#include "nodes/input.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mainera{

    MetricNode::MetricNode(const std::string &name, py::object py_func):
     Node(NodeType::METRIC, name, py_func) {}
    void MetricNode::execute(){

    }
}