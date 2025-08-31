#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/graph.hpp"
#include "core/node.hpp"

namespace py = pybind11;
using namespace aistudio;

PYBIND11_MODULE(graph, m) {
    m.doc() = "AI Studio C++ core bindings";

    py::class_<Node, Node::Ptr>(m, "Node")
        .def_property_readonly("name", [](const Node& n) { return n.name; })
        .def("clear_cache", &Node::clearCache);

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add", [](Graph &g, const std::string& name, py::function py_op, py::args args) {
            return g.add(name, py_op, args); // pass py::function directly
        })
        .def("add", [](Graph &g, const std::string& name) {
            return g.add(name);
        })
        .def("compute", [](Graph &g) -> py::object {
            PyObject* raw = static_cast<PyObject*>(g.compute());
            return py::reinterpret_borrow<py::object>(raw);
        });
}
