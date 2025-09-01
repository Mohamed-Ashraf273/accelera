#include "core/graph.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace aistudio {

Node::Ptr Graph::add(const std::string &name) {
  auto node = std::make_shared<Node>(name);
  nodes.push_back(node);
  return node;
}

Node::Ptr Graph::add(const std::string &name, py::object py_op, py::args args) {
  auto node = std::make_shared<Node>(name);

  // Detect if py_op is a model instance
  if (py::hasattr(py_op, "fit") && py::hasattr(py_op, "predict")) {
    node->setPyOp([py_op]() -> py::object { return py_op; });
  }
  // Check if this is a bound method (like model.predict)
  else if (py::hasattr(py_op, "__func__") && py::hasattr(py_op, "__self__")) {
    py::object method_func = py_op.attr("__func__");
    std::string method_name =
        py::str(method_func.attr("__name__")).cast<std::string>();

    node->setPyOpWithInput(
        [method_name, args](py::object fitted_model) -> py::object {
          return fitted_model.attr(method_name.c_str())(*args);
        });
  }
  // For regular functions
  else {
    node->setPyOp([py_op, args]() -> py::object { return py_op(*args); });
  }

  nodes.push_back(node);
  return node;
}

py::object Graph::compute(py::args args) {
  if (nodes.empty())
    return py::none();

  std::vector<py::object> node_results(nodes.size());

  for (size_t i = 0; i < nodes.size(); ++i) {

    if (nodes[i]->inputs.empty()) {
      // Node with no inputs (source node)
      if (i == 0) {
        // First node gets external args
        node_results[i] = py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject *>(nodes[i]->compute(args)));

        // If it's a model, fit it
        py::object result = node_results[i];
        if (py::hasattr(result, "fit") && py::len(args) >= 2) {
          result.attr("fit")(args[0], args[1]);
        }
      } else {
        // Other source nodes
        node_results[i] = py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject *>(nodes[i]->compute()));
      }
    } else {
      // Node with inputs - collect inputs from previous nodes
      // For single input nodes, use computeWithInput
      if (nodes[i]->inputs.size() == 1) {
        auto &input_node = nodes[i]->inputs[0];
        auto it = std::find(nodes.begin(), nodes.end(), input_node);
        if (it != nodes.end()) {
          size_t input_index = std::distance(nodes.begin(), it);
          void *input_ptr = node_results[input_index].ptr();
          node_results[i] =
              py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(
                  nodes[i]->computeWithInput(input_ptr)));
        }
      } else {
        // For multiple input nodes, use computeWithInputs
        std::vector<void *> input_ptrs;
        for (auto &input_node : nodes[i]->inputs) {
          auto it = std::find(nodes.begin(), nodes.end(), input_node);
          if (it != nodes.end()) {
            size_t input_index = std::distance(nodes.begin(), it);
            input_ptrs.push_back(node_results[input_index].ptr());
          }
        }

        node_results[i] =
            py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(
                nodes[i]->computeWithInputs(input_ptrs)));
      }
    }
  }

  return node_results.back();
}

void Graph::connect(Node::Ptr source, Node::Ptr target) {
  if (source && target) {
    source->outputs.push_back(target);
    target->inputs.push_back(source);
  }
}

void Graph::connect(const std::string &source_name,
                    const std::string &target_name) {
  Node::Ptr source, target;

  for (auto &node : nodes) {
    if (node->name == source_name)
      source = node;
    if (node->name == target_name)
      target = node;
  }

  if (source && target) {
    connect(source, target);
  }
}

void Graph::clearCache() {
  for (auto &node : nodes) {
    node->clearCache();
  }
}

} // namespace aistudio