#pragma once
#include <functional>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace aistudio {

// Forward declare Impl struct
struct NodeImpl;

class Node {
public:
  using Ptr = std::shared_ptr<Node>;

  Node(const std::string &name);
  ~Node();

  std::string name;
  py::object py_func;
  py::args args;
  Node::Ptr prev;
  Node::Ptr next;
  py::object input_value;
  py::object output_value;
  bool dirty = true;

  void setPyOp(std::function<py::object()> op);
  void setPyOpWithInput(std::function<py::object(py::object)> op);

  void clearCache();
  void compute(py::object input = py::none());

private:
  std::unique_ptr<NodeImpl> impl; // Use NodeImpl instead of Impl
};

} // namespace aistudio