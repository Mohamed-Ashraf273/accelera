#ifndef NODE_FACTORY_HPP
#define NODE_FACTORY_HPP

#include <memory>

#include "core/visibility.hpp"
#include "node.hpp"

#include <memory>

namespace accelera {

class ACCELERA_API NodeFactory {
public:
  static Node::Ptr createNode(NodeType type, const std::string &name,
                              py::object py_func);

  static Node::Ptr createNodeCopy(Node::Ptr node, int i);
};

} // namespace accelera

#endif // NODE_FACTORY_HPP