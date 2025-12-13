#ifndef PREPROCESS_NODE_HPP
#define PREPROCESS_NODE_HPP

#include "core/node.hpp"

namespace accelera {

class ACCELERA_API PreprocessNode : public Node {
public:
  PreprocessNode(const std::string &name, py::object py_func);
  void execute() override;

private:
  std::tuple<py::object, py::object> getInputData(std::shared_ptr<Node> input);
  void validateInputData(const py::object &X);
  std::tuple<py::object, py::object> processData(py::object X, py::object y);
  void storeResults(std::shared_ptr<Node> input, py::object processed_X,
                    py::object processed_y);
};

} // namespace accelera

#endif // PREPROCESS_NODE_HPP