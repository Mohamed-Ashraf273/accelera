#include "core/graph.hpp"
#include "core/node.hpp"
#include "core/node_factory.hpp"
#include "utils/graph_utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mainera;

PYBIND11_MODULE(graph, m) {
  m.doc() = "mAInera C++ core bindings";

  // Bind NodeType enum
  py::enum_<NodeType>(m, "NodeType")
      .value("INPUT", NodeType::INPUT)
      .value("MODEL", NodeType::MODEL)
      .value("PREDICT", NodeType::PREDICT)
      .value("METRIC", NodeType::METRIC)
      .value("FEATURE", NodeType::FEATURE)
      .value("PREPROCESS", NodeType::PREPROCESS)
      .value("MERGE", NodeType::MERGE)
      .export_values();

  // Node binding
  py::class_<Node, Node::Ptr>(m, "Node")
      .def_readonly("name", &Node::name)
      .def_readonly("type", &Node::type);

  // Graph binding with streamlined interface
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())

      .def("add_node", &Graph::add_node, py::arg("type"), py::arg("name"),
           py::arg("py_func"), "Add a node to the graph")

      .def("split", &Graph::split, py::arg("branch_name"),
           py::arg("branch_objects"), py::arg("node_types"),
           py::arg("node_names"), "Split the graph into branches")

      .def("execute", &Graph::execute, py::arg("X"), py::arg("y"),
           py::arg("best_path"),
           "Execute graph with inputs and return predictions")

      .def("mergeBranches", &Graph::mergeBranches, py::arg("merge_name"),
           py::arg("merge_func"), "Merge all branches back to sequential mode")

      .def("enableParallelExecution", &Graph::enableParallelExecution,
           py::arg("enable") = true, "Enable or disable parallel execution")

      .def("setMulticoreThreshold", &Graph::setMulticoreThreshold,
           py::arg("threshold"),
           "Set minimum number of tasks to use multicore execution")

      .def("clear", &Graph::clear, "Clear all nodes from the graph")

      .def("compile", &Graph::compile, "Compile the graph for execution")

      .def("get_nodes", &Graph::getNodes, "Get all nodes in the graph");

  // Utility functions
  m.def("serialize_graph", &serialize_graph, py::arg("graph"),
        py::arg("filepath"), "Serialize a graph structure to XML file");
}
