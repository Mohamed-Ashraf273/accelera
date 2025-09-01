#pragma once
#include <functional>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace aistudio {

class Node {
public:
  using Ptr = std::shared_ptr<Node>;

  Node(const std::string &name);
  ~Node();

  std::string name;
  std::vector<Ptr> inputs;
  std::vector<Ptr> outputs;
  bool dirty = true;

  // Store different types of operations
  void setPyOp(std::function<py::object()> op);
  void setPyOpWithInput(std::function<py::object(py::object)> op);
  void setPyOpWithInputs(std::function<py::object(py::args)> op);

  void clearCache();

  void *compute(const py::args &external_args = py::args());
  void *computeWithInput(void *input_ptr);
  void *computeWithInputs(const std::vector<void *> &input_ptrs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace aistudio