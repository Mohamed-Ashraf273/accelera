#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "core/graph.hpp"
#include "nodes/input.hpp"
#include "nodes/model.hpp"
#include "nodes/preprocess.hpp"

namespace py = pybind11;

namespace mainera {

ModelNode::ModelNode(const std::string &name, py::object py_func)
    : Node(NodeType::MODEL, name, py_func) {

  if (!py::hasattr(py_func["model"], "fit")) {
    throw std::runtime_error("Model node '" + name +
                             "' must have a 'fit' method");
  }
}

void ModelNode::execute() {
  if (getGraph()->getIsExecuted()) {
    return;
  }
  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Model node '" + name +
                               "' requires a valid input node");
    }

    std::shared_ptr<InputNode> data;

    if (input->type == NodeType::INPUT) {
      auto input_node = std::dynamic_pointer_cast<InputNode>(input);
      if (!input_node) {
        throw std::runtime_error("Failed to cast to InputNode");
      }
      data = input_node;
    } else {
      auto preprocess_node = std::dynamic_pointer_cast<PreprocessNode>(input);
      if (!preprocess_node) {
        throw std::runtime_error("Failed to cast to PreprocessNode");
      }
      data = preprocess_node->getData();
    }

    if (!data) {
      throw std::runtime_error("No valid data source found");
    }

    py::object X = data->getX();
    std::string model_type = py_func["model_type"].cast<std::string>();

    if (X.is_none()) {
      throw std::runtime_error("Model node '" + name +
                               "' received None inputs (X)");
    }

    py::object model_instance =
        py::module::import("copy").attr("deepcopy")(py_func["model"]);

    try {
      if (model_type == "supervised") {
        py::object y = data->getY();
        if (y.is_none()) {
          throw std::runtime_error("Model node '" + name +
                                   "' received None inputs (y)");
        }
        if (!py::hasattr(X, "shape") || !py::hasattr(y, "shape")) {
          throw std::runtime_error("Model node '" + name +
                                   "' inputs must be array-like objects");
        }
        model_instance.attr("fit")(X, y);
      } else if (model_type == "unsupervised") {
        if (!py::hasattr(X, "shape")) {
          throw std::runtime_error("Model node '" + name +
                                   "' inputs must be array-like objects");
        }
        model_instance.attr("fit")(X);
      } else {
        throw std::runtime_error("Model node '" + name +
                                 "' has unknown model type");
      }
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in model fitting: " +
                               std::string(e.what()));
    }
    setData(model_instance);
  } catch (const py::error_already_set &e) {
    throw std::runtime_error("ModelNode: Python error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("ModelNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace mainera