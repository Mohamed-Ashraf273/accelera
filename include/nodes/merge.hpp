#ifndef MERGE_HPP
#define MERGE_HPP

#include "core/node.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

class MergeNode : public Node {
public:
  MergeNode(const std::string &name, size_t numInputs, size_t numOutputs,
            py::object py_func);

  void execute() override;

  // Get the result of the merge operation
  py::object getResult() const { return m_result; }

private:
  py::object m_result = py::none(); // Store the merge result
};

} // namespace aistudio

#endif // MERGE_HPP
