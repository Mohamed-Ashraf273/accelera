#include "core/graph.hpp"
#include "core/node.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace aistudio;

PYBIND11_MODULE(graph, m) {
  m.doc() = "AI Studio C++ core bindings";

  // Node binding
  py::class_<Node, Node::Ptr>(m, "Node")
      .def(py::init<const std::string &>())
      .def_property_readonly("name", [](const Node &n) { return n.name; })
      .def("set_py_op", &Node::setPyOp)
      .def("set_py_op_with_input", &Node::setPyOpWithInput)
      .def("set_py_op_with_inputs", &Node::setPyOpWithInputs)
      .def("clear_cache", &Node::clearCache)
      .def("compute",
           [](Node &n, py::args args) -> py::object {
             return py::reinterpret_borrow<py::object>(
                 reinterpret_cast<PyObject *>(n.compute(args)));
           })
      .def("compute_with_input",
           [](Node &n, py::object input) -> py::object {
             return py::reinterpret_borrow<py::object>(
                 reinterpret_cast<PyObject *>(n.computeWithInput(input.ptr())));
           })
      .def("compute_with_inputs", [](Node &n, py::list inputs) -> py::object {
        std::vector<void *> input_ptrs;
        for (auto &item : inputs) {
          input_ptrs.push_back(item.ptr());
        }
        return py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject *>(n.computeWithInputs(input_ptrs)));
      });

  // Graph binding
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def("add", [](Graph &g, const std::string &name, py::object py_op,
                     py::args args) { return g.add(name, py_op, args); })
      .def("add", [](Graph &g, const std::string &name) { return g.add(name); })
      .def("connect", [](Graph &g, Node::Ptr source,
                         Node::Ptr target) { g.connect(source, target); })
      .def("connect",
           [](Graph &g, const std::string &source_name,
              const std::string &target_name) {
             g.connect(source_name, target_name);
           })
      .def(
          "compute",
          [](Graph &g, py::args args) -> py::object { return g.compute(args); })
      .def("clear_cache", &Graph::clearCache);
}