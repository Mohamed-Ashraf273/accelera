#include "core/node.hpp"

#include "core/graph.hpp"
#include "nodes/input.hpp"

#include <stdexcept>

namespace mainera {

Node::Node(NodeType type, const std::string &name, py::object py_func)
    : type(type), name(name), py_func(py_func) {}
Node::~Node() {
  // Cleanup handled by shared_ptr
}

void Node::setSourceNode(std::shared_ptr<Node> source) {
  m_sourceNode = source;
}

std::shared_ptr<Node> Node::getSourceNode() const { return m_sourceNode; }

void Node::setData(py::object result) { m_data = result; }

py::object Node::getData() const { return m_data; }

void Node::setGraph(Graph *graph) { m_graph = graph; }

Graph *Node::getGraph() const { return m_graph; }

void Node::setShouldCreateNewData(bool should_create) {
  should_create_new_data = should_create;
}

bool Node::getShouldCreateNewData() const { return should_create_new_data; }

void Node::setUsesGPU(bool uses_gpu) { m_uses_gpu = uses_gpu; }

bool Node::getUsesGPU() const { return m_uses_gpu; }

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