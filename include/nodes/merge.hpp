#ifndef MERGE_NODE_HPP
#define MERGE_NODE_HPP

#include "core/node.hpp"

namespace aistudio {

class MergeNode : public Node {
public:
  MergeNode(const std::string &name, size_t numInputs, size_t numOutputs,
            py::object py_func);
  void execute() override;
};

} // namespace aistudio

#endif // MERGE_NODE_HPP
