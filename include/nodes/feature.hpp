#ifndef FEATURE_NODE_HPP
#define FEATURE_NODE_HPP

#include "core/node.hpp"

namespace mainera {

class MAINERA_API FeatureNode : public Node {
public:
  FeatureNode(const std::string &name, py::object py_func);
  void execute() override;
};

} // namespace mainera

#endif // FEATURE_NODE_HPP