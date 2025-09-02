#include "core/pipeline.hpp"
#include "core/node.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace aistudio;

PYBIND11_MODULE(pipeline, m) {
  m.doc() = "AI Studio C++ core bindings";

  // Node binding
  py::class_<Node, Node::Ptr>(m, "Node")
      .def(py::init<const std::string &>())
      .def_readonly("name", &Node::name);

  // Pipeline binding
  py::class_<Pipeline>(m, "Pipeline")
      .def(py::init<>())
      .def("add", [](Pipeline &g, const std::string &name, py::object py_op,
                     py::args args) { return g.add(name, py_op, args); })
      .def("compute", [](Pipeline &g, py::args args) -> py::object {
        return g.compute(args);
      });
}