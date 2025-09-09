#ifndef MODEL_NODE_HPP
#define MODEL_NODE_HPP

#include "core/node.hpp"

namespace mainera {

class MAINERA_API ModelNode : public Node {
public:
  ModelNode(const std::string &name, py::object py_func);
  void execute() override;
};

} // namespace mainera

#endif // MODEL_NODE_HPP