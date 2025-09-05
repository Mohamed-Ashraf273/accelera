#ifndef FEATURE_NODE_HPP
#define FEATURE_NODE_HPP

#include "core/node.hpp"

namespace mainera {

class FeatureNode : public Node {
public:
  FeatureNode(const std::string &name, size_t numInputs, size_t numOutputs,
              py::object py_func);
  void execute() override;
};

} // namespace mainera

#endif // FEATURE_NODE_HPP