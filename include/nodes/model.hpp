#ifndef MODEL_NODE_HPP
#define MODEL_NODE_HPP

#include "core/node.hpp"

namespace aistudio {

class ModelNode : public Node {
public:
  ModelNode(const std::string &name, size_t numInputs, size_t numOutputs,
            py::object py_func);
  void execute() override;
};

} // namespace aistudio

#endif // MODEL_NODE_HPP