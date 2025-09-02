#ifndef NODE_FACTORY_HPP
#define NODE_FACTORY_HPP

#include "node.hpp"
#include <memory>

namespace aistudio {

class NodeFactory {
public:
  static Node::Ptr createNode(NodeType type, const std::string &name,
                              size_t numInputs, size_t numOutputs,
                              py::object py_func);
};

} // namespace aistudio

#endif // NODE_FACTORY_HPP