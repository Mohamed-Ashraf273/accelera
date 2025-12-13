#ifndef MERGE_HPP
#define MERGE_HPP

#include <pybind11/pybind11.h>

#include "core/node.hpp"

namespace py = pybind11;

namespace accelera {

class ACCELERA_API MergeNode : public Node {
public:
  MergeNode(const std::string &name, py::object py_func);
  void execute() override;
};

} // namespace accelera

#endif // MERGE_HPP
