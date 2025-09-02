#pragma once
#include "core/node.hpp"
#include <functional>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace aistudio {

class Pipeline {
public:
  Pipeline() : head(nullptr), tail(nullptr) {}

  // Add a Python callable node (function or model)
  Node::Ptr add(const std::string &name, py::object py_op,
                py::args args = py::args());

  // Compute final node with optional input arguments
  py::object compute(py::args args = py::args());

private:
  Node::Ptr head;
  Node::Ptr tail; // Add tail pointer
};

} // namespace aistudio