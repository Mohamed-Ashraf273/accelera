#pragma once
#include <string>
#include <vector>
#include <memory>
#include "core/node.hpp"

// Include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/functional.h> 

namespace py = pybind11;

namespace aistudio {

class Graph {
public:
    using Ptr = std::shared_ptr<Graph>;

    Graph() = default;

    // Add node without Python callable
    Graph& add(const std::string& name);

    // Add node with Python callable and arguments
    Graph& add(const std::string& name, py::function py_op, py::args args);

    void* compute();  // returns raw PyObject* (last node's output)
    void clearCache();

private:
    std::vector<Node::Ptr> nodes;
};

} // namespace aistudio
