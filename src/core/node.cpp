#include "core/node.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace aistudio {

struct NodeImpl {
  std::function<py::object()> py_op = nullptr;
  std::function<py::object(py::object)> py_op_with_input = nullptr;
};

Node::Node(const std::string &n)
    : name(n), impl(std::make_unique<NodeImpl>()) {}

Node::~Node() = default;

void Node::setPyOp(std::function<py::object()> op) {
  impl->py_op = std::move(op);
}

void Node::setPyOpWithInput(std::function<py::object(py::object)> op) {
  impl->py_op_with_input = std::move(op);
}

void Node::clearCache() {
  dirty = true;
  input_value = py::none();
  output_value = py::none();

  Node::Ptr next_node = next;
  while (next_node) {
    next_node->clearCache();
    next_node = next_node->next;
  }
}

void Node::compute(py::object input) {
  if (!dirty && !output_value.is_none()) {
    return;
  }

  if (!input.is_none()) {
    input_value = input;
  }

  try {
    // Case 1: Model instance - just return it, don't call it
    if (py_func && py::hasattr(py_func, "fit")) {
      output_value = py_func;
    }
    // Case 2: Pre-defined operation with input
    else if (impl->py_op_with_input && !input_value.is_none()) {
      output_value = impl->py_op_with_input(input_value);
    }
    // Case 3: Pre-defined operation without input
    else if (impl->py_op) {
      output_value = impl->py_op();
    }
    // Case 4: Python function with stored args (priority)
    else if (py_func && !py_func.is_none() && args && py::len(args) > 0) {
      output_value = py_func(*args);
    }
    // Case 5: Python function with input
    else if (py_func && !py_func.is_none() && !input_value.is_none()) {
      // Handle bound methods (like model.predict)
      if (py::hasattr(py_func, "__func__") &&
          py::hasattr(py_func, "__self__")) {
        py::object self = py_func.attr("__self__");
        std::string method_name =
            py::str(py_func.attr("__func__").attr("__name__"))
                .cast<std::string>();

        // Use the fitted model from previous node if available
        if (prev && py::hasattr(prev->output_value, "fit")) {
          self = prev->output_value;
        } else if (!py::hasattr(self, "fit")) {
          throw std::runtime_error("Model must be fitted before calling " +
                                   method_name);
        }

        if (input_value.is(prev->output_value)) {
          // Check if prev exists and has valid input_value
          if (prev && !prev->input_value.is_none()) {
            // Check if input_value is a tuple with elements
            if (py::isinstance<py::tuple>(prev->input_value)) {
              py::tuple input_tuple = prev->input_value.cast<py::tuple>();
              if (py::len(input_tuple) > 0) {
                input_value = input_tuple[0];
              }
            }
            // Check if input_value is a list with elements
            else if (py::isinstance<py::list>(prev->input_value)) {
              py::list input_list = prev->input_value.cast<py::list>();
              if (py::len(input_list) > 0) {
                input_value = input_list[0];
              }
            } else {
              input_value = prev->input_value;
            }
          }
        }

        output_value = self.attr(method_name.c_str())(input_value);
      }
      // Handle regular functions with input
      else {
        output_value = py_func(input_value);
      }
    }
    // Case 6: Python function without input or args
    else if (py_func && !py_func.is_none()) {
      output_value = py_func();
    }
    // Case 7: No operation defined
    else {
      output_value = py::none();
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Exception in Node::compute: ") +
                             e.what());
  }
  dirty = false;
}

} // namespace aistudio