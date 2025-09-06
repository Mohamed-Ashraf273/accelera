#include "nodes/merge.hpp"
#include "nodes/input.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

MergeNode::MergeNode(const std::string &name, py::object py_func)
    : Node(NodeType::MERGE, name, py_func) {}

void MergeNode::execute() {
  try {
    std::shared_ptr<InputNode> input = getInput();

    if (!input) {
      throw std::runtime_error("Merge node '" + name +
                               "' requires a valid input");
    }

    py::object X = input->getX();
    py::object y = input->getY();
    py::object fitted_model = input->getFittedModel();

    py::object result;
    try {
      result = py_func(y);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in merge function: " +
                               std::string(e.what()));
    }

    auto new_input = std::make_shared<InputNode>();
    new_input->setInputData(X, result);
    new_input->setFittedModel(fitted_model);

    m_result = result;

    setOutput(new_input);

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("MergeNode: Error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("MergeNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace mainera
