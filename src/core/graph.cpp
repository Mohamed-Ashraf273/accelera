#include "core/graph.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

namespace aistudio {

Graph& Graph::add(const std::string& name) {
    auto node = std::make_shared<Node>(name);

    if (!nodes.empty()) {
        nodes.back()->outputs.push_back(node);
        node->inputs.push_back(nodes.back());
    }

    nodes.push_back(node);
    return *this;
}

Graph& Graph::add(const std::string& name, py::function py_op, py::args args) {
    auto node = std::make_shared<Node>(name);

    // Store lambda capturing Python callable + args
    node->setPyOp([py_op, args]() -> py::object {
        return py_op(*args);  // call with unpacked args and return result
    });

    if (!nodes.empty()) {
        nodes.back()->outputs.push_back(node);
        node->inputs.push_back(nodes.back());
    }
    nodes.push_back(node);
    return *this;
}


void* Graph::compute() {
    if (nodes.empty()) return nullptr;
    return nodes.back()->compute();  // returns PyObject* from last node
}

void Graph::clearCache() {
    for (auto& node : nodes) {
        node->clearCache();
    }
}

} // namespace aistudio
