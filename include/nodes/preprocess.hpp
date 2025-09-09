#ifndef PREPROCESS_NODE_HPP
#define PREPROCESS_NODE_HPP

#include "core/node.hpp"

namespace mainera {

class MAINERA_API PreprocessNode : public Node {
public:
  PreprocessNode(const std::string &name, py::object py_func);
  void execute() override;
};

} // namespace mainera

#endif // PREPROCESS_NODE_HPP