#ifndef MODEL_NODE_HPP
#define MODEL_NODE_HPP

#include "core/node.hpp"

namespace accelera {

class ACCELERA_API ModelNode : public Node {
public:
  ModelNode(const std::string &name, py::object py_func);
  void execute() override;

private:
  std::tuple<py::object, py::object> getInputData(std::shared_ptr<Node> input);
  void validateInputData(const py::object &X, const py::object &y);
  py::object fitModel(py::object X, py::object y);
};

} // namespace accelera

#endif // MODEL_NODE_HPP