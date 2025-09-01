#pragma once
#include "core/node.hpp"
#include <functional>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace aistudio {

class Graph {
public:
  using Ptr = std::shared_ptr<Graph>;

  Graph() = default;

  // Add a regular node
  Node::Ptr add(const std::string &name);

  // Add a Python callable node (function or model)
  Node::Ptr add(const std::string &name, py::object py_op, py::args args);

  // Connect nodes
  void connect(Node::Ptr source, Node::Ptr target);
  void connect(const std::string &source_name, const std::string &target_name);

  // Compute final node with optional input arguments
  py::object compute(py::args args = py::args());

  void clearCache();

private:
  std::vector<Node::Ptr> nodes;
};

} // namespace aistudio