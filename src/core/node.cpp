#include "core/node.hpp"
#include "core/graph.hpp"
#include "nodes/input.hpp"
#include <iostream>
#include <stdexcept>

namespace mainera {

Node::Node(NodeType type, const std::string &name, py::object py_func)
    : type(type), name(name), py_func(py_func) {}
Node::~Node() {
  // Cleanup handled by shared_ptr
}

std::shared_ptr<InputNode> Node::getInput() { return m_sourceNode->m_data; }

void Node::setOutput(std::shared_ptr<InputNode> result) { m_data = result; }

std::shared_ptr<InputNode> Node::getOutput() { return m_data; }

void Node::setGraph(Graph *graph) { m_graph = graph; }

Graph *Node::getGraph() const { return m_graph; }

void Node::setShouldCreateNewData(bool should_create) {
  should_create_new_data = should_create;
}

bool Node::getShouldCreateNewData() const { return should_create_new_data; }

void Node::usesGPU() {
  try {
    py::object model = py_func;

    if (py::hasattr(model, "device")) {
      py::object device_attr = model.attr("device");
      std::string device_str = device_attr.attr("type").cast<std::string>();
      setUsesGPU(device_str == "cuda");
    }

    if (py::hasattr(model, "model")) {
      py::object inner_model = model.attr("model");
      if (py::hasattr(inner_model, "device")) {
        py::object device_attr = inner_model.attr("device");
        std::string device_str = device_attr.attr("type").cast<std::string>();
        setUsesGPU(device_str == "cuda");
      }
    }

  } catch (const py::error_already_set &e) {
    PyErr_Clear();
  }
}

} // namespace mainera