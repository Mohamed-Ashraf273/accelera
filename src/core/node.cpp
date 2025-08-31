#include "core/node.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

namespace aistudio {

struct __attribute__((visibility("hidden"))) Node::Impl {
    std::function<py::object()> py_op = nullptr;
    py::object cache = py::none();
};

Node::Node(const std::string& n)
    : name(n), impl(std::make_unique<Impl>()) {}

Node::~Node() = default;

// Store lambda or Python callable
void Node::setPyOp(std::function<py::object()> op) {
    impl->py_op = std::move(op);
}

void Node::clearCache() {
    dirty = true;
    impl->cache = py::none();

    for (auto& out : outputs) {
        if (out) out->clearCache();
    }
}

void* Node::compute() {
    if (!dirty && !impl->cache.is_none()) {
        return impl->cache.ptr();
    }

    std::vector<py::object> args;
    for (auto& inp : inputs) {
        args.push_back(py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject*>(inp->compute())
        ));
    }

    if (impl->py_op) {
        if (args.empty()) {
            impl->cache = impl->py_op();
        } else if (args.size() == 1) {
            impl->cache = impl->py_op();
        } else {
            // pass tuple of arguments
            impl->cache = impl->py_op();
        }
    }

    dirty = false;
    return impl->cache.ptr();
}

} // namespace aistudio
