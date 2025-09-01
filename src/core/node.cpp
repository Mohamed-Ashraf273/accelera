#include "core/node.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace aistudio {

struct Node::Impl {
  std::function<py::object()> py_op = nullptr;
  std::function<py::object(py::object)> py_op_with_input = nullptr;
  std::function<py::object(py::args)> py_op_with_inputs = nullptr;
  py::object cache = py::none();
};

Node::Node(const std::string &n) : name(n), impl(std::make_unique<Impl>()) {}

Node::~Node() = default;

void Node::setPyOp(std::function<py::object()> op) {
  impl->py_op = std::move(op);
}

void Node::setPyOpWithInput(std::function<py::object(py::object)> op) {
  impl->py_op_with_input = std::move(op);
}

void Node::setPyOpWithInputs(std::function<py::object(py::args)> op) {
  impl->py_op_with_inputs = std::move(op);
}

void Node::clearCache() {
  dirty = true;
  impl->cache = py::none();

  for (auto &out : outputs) {
    if (out)
      out->clearCache();
  }
}

void *Node::compute(const py::args &external_args) {
  if (!dirty && !impl->cache.is_none())
    return impl->cache.ptr();

  if (impl->py_op) {
    impl->cache = impl->py_op();
  }

  dirty = false;
  return impl->cache.ptr();
}

void *Node::computeWithInput(void *input_ptr) {
  if (!dirty && !impl->cache.is_none())
    return impl->cache.ptr();
  if (impl->py_op_with_input && input_ptr) {
    py::object input = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(input_ptr));
    try {
      impl->cache = impl->py_op_with_input(input);
    } catch (const std::exception &e) {
      impl->cache = py::none();
    }
  } else {
    impl->cache = py::none();
  }

  dirty = false;
  return impl->cache.ptr();
}

void *Node::computeWithInputs(const std::vector<void *> &input_ptrs) {
  if (!dirty && !impl->cache.is_none())
    return impl->cache.ptr();

  if (impl->py_op_with_inputs && !input_ptrs.empty()) {
    py::tuple inputs(input_ptrs.size());
    for (size_t i = 0; i < input_ptrs.size(); ++i) {
      inputs[i] = py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject *>(input_ptrs[i]));
    }

    try {
      impl->cache = impl->py_op_with_inputs(inputs);
    } catch (const std::exception &e) {
      std::cerr << "Exception in Node::computeWithInputs: " << e.what()
                << std::endl;
      impl->cache = py::none();
    }
  } else {
    impl->cache = py::none();
  }

  dirty = false;
  return impl->cache.ptr();
}

} // namespace aistudio