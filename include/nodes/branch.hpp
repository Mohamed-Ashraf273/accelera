#ifndef BRANCH_NODE_HPP
#define BRANCH_NODE_HPP

#include "core/node.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

class BranchNode : public Node {
public:
  BranchNode(const std::string &name, size_t numInputs, size_t numOutputs,
             py::object py_func);
  void execute() override;
};

} // namespace aistudio

#endif // BRANCH_NODE_HPP