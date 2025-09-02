#ifndef PREPROCESS_NODE_HPP
#define PREPROCESS_NODE_HPP

#include "core/node.hpp"

namespace aistudio {

class PreprocessNode : public Node {
public:
  PreprocessNode(const std::string &name, size_t numInputs, size_t numOutputs,
                 py::object py_func);
  void execute() override;
};

} // namespace aistudio

#endif // PREPROCESS_NODE_HPP