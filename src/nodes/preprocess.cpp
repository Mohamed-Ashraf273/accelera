#include "nodes/preprocess.hpp"
#include "nodes/input.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

PreprocessNode::PreprocessNode(const std::string &name, py::object py_func)
    : Node(NodeType::PREPROCESS, name, py_func) {}

void PreprocessNode::setData(std::shared_ptr<InputNode> input) {
  m_input = input;
}

std::shared_ptr<InputNode> PreprocessNode::getData() const { return m_input; }

void PreprocessNode::execute() {
  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Preprocess node '" + name +
                               "' requires a valid input");
    }

    std::shared_ptr<InputNode> data;

    if (input->type != NodeType::INPUT) {
      auto preprocess_input = std::dynamic_pointer_cast<PreprocessNode>(input);
      if (preprocess_input) {
        data = preprocess_input->getData();
      } else {
        throw std::runtime_error(
            "Preprocess node '" + name +
            "' requires an InputNode or PreprocessNode as input");
      }
    } else {
      data = std::dynamic_pointer_cast<InputNode>(input);
      if (!data) {
        throw std::runtime_error("Failed to cast to InputNode");
      }
    }

    if (!data) {
      throw std::runtime_error("No valid data source found");
    }

    py::object X = data->getX();
    py::object y = data->getY();

    if (should_create_new_data) {
      py::object X_copy = X.attr("copy")();
      py::object y_copy = y.is_none() ? py::none() : y.attr("copy")();

      if (X_copy.is_none()) {
        throw std::runtime_error("Preprocess node '" + name +
                                 "' received None for X input");
      }

      try {
        X_copy = py_func(X_copy);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in preprocessing: " +
                                 std::string(e.what()));
      }

      auto new_input = std::make_shared<InputNode>();
      new_input->setInputData(X_copy, y_copy);
      setData(new_input);
    } else {
      if (X.is_none()) {
        throw std::runtime_error("Preprocess node '" + name +
                                 "' received None for X input");
      }

      try {
        X = py_func(X);
      } catch (const py::error_already_set &e) {
        throw std::runtime_error("Python error in preprocessing: " +
                                 std::string(e.what()));
      }

      data->setInputData(X, y);
      setData(data);
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in preprocess node '" + name +
                             "': " + e.what());
  }
}

} // namespace mainera