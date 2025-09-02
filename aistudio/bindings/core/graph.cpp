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

  // Graph binding with automatic connection management
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(
          "add_node",
          [](Graph &g, NodeType type, const std::string &name,
             py::object py_func, size_t num_inputs = 1,
             size_t num_outputs = 1) -> Node::Ptr {
            // Create node using factory
            auto node = NodeFactory::createNode(type, name, num_inputs,
                                                num_outputs, py_func);

            // Add to graph
            g.addNode(node);

            return node;
          },
          py::arg("type"), py::arg("name"), py::arg("py_func"),
          py::arg("num_inputs") = 1, py::arg("num_outputs") = 1,
          "Add a node to the graph with automatic connection management")

      .def(
          "connect",
          [](Graph &g, Node::Ptr source, size_t source_output, Node::Ptr target,
             size_t target_input) {
            source->connectTo(source_output, target, target_input);
          },
          py::arg("source"), py::arg("source_output"), py::arg("target"),
          py::arg("target_input"), "Manually connect two nodes")

      .def("auto_connect", &Graph::autoConnectNodes,
           "Automatically connect nodes based on their types and order")

      .def("compile", &Graph::compile,
           "Finalize graph structure and prepare for execution")

      .def("run", &Graph::run, "Execute the entire graph")

      .def("execute", &Graph::execute,
           "Execute graph with inputs and return predictions", py::arg("X"),
           py::arg("y"))

      .def(
          "set_input",
          [](Graph &g, Node::Ptr node, size_t input_index, py::object data) {
            // Set external input data for a node
            if (input_index < node->getInputEdges().size()) {
              node->getInputEdges()[input_index]->setData(data);
            } else {
              throw py::index_error("Input index out of range for node: " +
                                    node->name);
            }
          },
          py::arg("node"), py::arg("input_index"), py::arg("data"),
          "Set input data for a specific node")

      .def(
          "get_output",
          [](Graph &g, Node::Ptr node, size_t output_index) -> py::object {
            // Get output data from a node
            if (output_index < node->getOutputEdges().size() &&
                node->getOutputEdges()[output_index]->isReady()) {
              return node->getOutputEdges()[output_index]->getData();
            }
            return py::none();
          },
          py::arg("node"), py::arg("output_index") = 0,
          "Get output data from a specific node")

      .def("get_nodes", &Graph::getNodes, "Get all nodes in the graph")

      .def("clear", &Graph::clear, "Clear all nodes from the graph")

      .def("serialize", &Graph::serialize, py::arg("filepath"),
           "Serialize the graph structure to XML file for visualization")

      // Parallel execution methods
      .def("enableParallelExecution", &Graph::enableParallelExecution,
           py::arg("enable") = true, "Enable or disable parallel execution")

      .def("createBranch", &Graph::createBranch, py::arg("name"),
           py::arg("models"), py::arg("merge_func") = py::none(),
           "Create a branch node that splits input to multiple models")

      .def("createBranchPipeline", &Graph::createBranchPipeline,
           py::arg("branch_name"), py::arg("models"), py::arg("predict_name"),
           py::arg("test_data"),
           "Create complete branch pipeline with branch node, models, and "
           "predict nodes")

      .def("createPredictForBranches", &Graph::createPredictForBranches,
           py::arg("predict_name"), py::arg("models"),
           py::arg("test_data") = py::none(),
           "Create predict nodes for existing branches")

      .def("duplicateNodeForBranches", &Graph::duplicateNodeForBranches,
           py::arg("template_node"), py::arg("branch_models"),
           "Duplicate a node for each branch path")

      .def("collectBranchResults", &Graph::collectBranchResults,
           py::arg("result_nodes"),
           "Collect results from multiple branch result nodes")

      // Pipeline management methods - all complex logic in C++
      .def("addSequentialNode", &Graph::addSequentialNode, py::arg("node_type"),
           py::arg("name"), py::arg("obj"),
           "Add node in sequential mode with automatic connection")
      .def("startBranching", &Graph::startBranching, py::arg("branch_name"),
           py::arg("branch_models"), "Start branching mode from current node")
      .def("addToBranches", &Graph::addToBranches, py::arg("node_type"),
           py::arg("name"), py::arg("obj"), "Add node to all current branches")
      .def("mergeBranches", &Graph::mergeBranches, py::arg("merge_name"),
           py::arg("merge_func"), "Merge all branches back to sequential mode")
      .def("isBranched", &Graph::isBranched,
           "Check if currently in branched mode");

  m.def("create_model_node", [](const std::string &name, py::object py_func) {
    return NodeFactory::createNode(NodeType::MODEL, name, 1, 1, py_func);
  });

  m.def("create_predict_node", [](const std::string &name, py::object py_func) {
    return NodeFactory::createNode(NodeType::PREDICT, name, 2, 1,
                                   py_func); // Typically takes model + data
  });

  m.def("create_feature_node", [](const std::string &name, py::object py_func) {
    return NodeFactory::createNode(NodeType::FEATURE, name, 1, 1, py_func);
  });

  m.def("create_preprocess_node", [](const std::string &name,
                                     py::object py_func) {
    return NodeFactory::createNode(NodeType::PREPROCESS, name, 1, 1, py_func);
  });
}