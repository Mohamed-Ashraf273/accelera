#ifndef NODE_FACTORY_HPP
#define NODE_FACTORY_HPP

#include "core/visibility.hpp"
#include "node.hpp"
#include <memory>

namespace mainera {

class MAINERA_API NodeFactory {
public:
  static Node::Ptr createNode(NodeType type, const std::string &name,
                              py::object py_func);
};

} // namespace mainera

#endif // NODE_FACTORY_HPP