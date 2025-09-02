#ifndef PREDICT_NODE_HPP
#define PREDICT_NODE_HPP

#include "core/node.hpp"

namespace aistudio {

class PredictNode : public Node {
public:
  PredictNode(const std::string &name, size_t numInputs, size_t numOutputs,
              py::object py_func);
  void execute() override;
};

} // namespace aistudio

#endif // PREDICT_NODE_HPP