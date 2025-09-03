#include "core/graph.hpp"
#include "core/node.hpp"
#include "core/node_factory.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace aistudio;

PYBIND11_MODULE(graph, m) {
  m.doc() = "AI Studio C++ core bindings";

  // Bind NodeType enum
  py::enum_<NodeType>(m, "NodeType")
      .value("MODEL", NodeType::MODEL)
      .value("PREDICT", NodeType::PREDICT)
      .value("FEATURE", NodeType::FEATURE)
      .value("PREPROCESS", NodeType::PREPROCESS)
      .value("BRANCH", NodeType::BRANCH)
      .export_values();

  // Node binding
  py::class_<Node, Node::Ptr>(m, "Node")
      .def_readonly("name", &Node::name)
      .def_readonly("type", &Node::type)
      .def_readonly("dirty", &Node::dirty)
      .def("connectTo", &Node::connectTo, py::arg("myOutputIndex"),
           py::arg("targetNode"), py::arg("targetInputIndex"),
           "Connect this node's output to target node's input");

  // Graph binding with streamlined interface
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())

      .def("add_node", &Graph::add_node, py::arg("type"), py::arg("name"),
           py::arg("py_func"), py::arg("num_inputs") = 1,
           py::arg("num_outputs") = 1, "Add a node to the graph")

      .def("compile", &Graph::compile, "Compile the graph for execution")

      .def("execute", &Graph::execute, py::arg("X"), py::arg("y"),
           "Execute graph with inputs and return predictions")

      .def("clear", &Graph::clear, "Clear all nodes from the graph")

      .def("serialize", &Graph::serialize, py::arg("filepath"),
           "Serialize the graph structure to XML file")

      .def("enableParallelExecution", &Graph::enableParallelExecution,
           py::arg("enable") = true, "Enable or disable parallel execution")

      .def("setMulticoreThreshold", &Graph::setMulticoreThreshold,
           py::arg("threshold"),
           "Set minimum number of tasks to use multicore execution")

      .def("get_nodes", &Graph::getNodes, "Get all nodes in the graph")

      // Pipeline management methods
      .def("addSequentialNode", &Graph::addSequentialNode, py::arg("node_type"),
           py::arg("name"), py::arg("obj"), "Add node in sequential mode")
      .def("startBranching", &Graph::startBranching, py::arg("branch_name"),
           py::arg("branch_models"), "Start branching mode from current node")
      .def("startBranchingWithTypes", &Graph::startBranchingWithTypes,
           py::arg("branch_name"), py::arg("branch_objects"),
           py::arg("node_types"), py::arg("node_names"),
           "Start branching with explicit node types")
      .def("addToBranches", &Graph::addToBranches, py::arg("node_type"),
           py::arg("name"), py::arg("obj"), "Add node to all current branches")
      .def("mergeBranches", &Graph::mergeBranches, py::arg("merge_name"),
           py::arg("merge_func"), "Merge all branches back to sequential mode")
      .def("isBranched", &Graph::isBranched,
           "Check if currently in branched mode");
}