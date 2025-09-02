#include "core/pipeline.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace aistudio {

Node::Ptr Pipeline::add(const std::string &name, py::object py_op,
                        py::args args) {
  auto node = std::make_shared<Node>(name);
  node->py_func = py_op;
  node->args = args;

  if (head == nullptr) {
    head = node;
    tail = node;
    node->prev = nullptr;
  } else {
    tail->next = node;
    node->prev = tail;
    tail = node;
  }

  return node;
}

py::object Pipeline::compute(py::args args) {
  if (head == nullptr) {
    return py::none();
  }

  head->clearCache();

  Node::Ptr current = head;
  while (current != nullptr) {

    if (current == head) {
      current->compute(args);

      // Auto-fit models if this is a model node and training data is provided
      if (current->py_func && py::hasattr(current->py_func, "fit") &&
          py::len(args) >= 2) {
        try {
          current->py_func.attr("fit")(args[0], args[1]);
          current->output_value = current->py_func;
        } catch (const std::exception &e) {
          throw std::runtime_error(std::string("Exception fitting model: ") +
                                   e.what());
        }
      }
    } else {
      current->compute(current->prev->output_value);
    }

    if (current->next) {
      current = current->next;
    } else {
      current = nullptr;
    }
  }
  return tail->output_value;
}

} // namespace aistudio