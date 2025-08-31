#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

class Node {
public:
    using Ptr = std::shared_ptr<Node>;

    std::string name;
    std::vector<Ptr> inputs;
    std::vector<Ptr> outputs;
    bool dirty = true;

    Node(const std::string& n);
    ~Node();

    // Store a callable (Python function or C++ lambda)
    void setPyOp(std::function<py::object()> op);
    void clearCache();

    // compute returns a PyObject* (void* for opaque)
    void* compute();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace aistudio
