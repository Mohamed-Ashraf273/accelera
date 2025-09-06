#include "core/node.hpp"
#include "core/graph.hpp"
#include "nodes/input.hpp"
#include <iostream>
#include <stdexcept>

namespace mainera {

Node::Node(NodeType type, const std::string &name, py::object py_func)
    : type(type), name(name), py_func(py_func) {

  // Initialize single input and output edges
  m_inputEdge = std::make_shared<Edge>();
  m_outputEdge = std::make_shared<Edge>();
}

Node::~Node() {
  // Cleanup handled by shared_ptr
}

std::shared_ptr<InputNode> Node::getInput() {
  if (m_inputEdge) {
    return m_inputEdge->getData();
  }
  return nullptr;
}

void Node::setOutput(std::shared_ptr<InputNode> result) {
  if (m_outputEdge) {
    m_outputEdge->setData(result);
  }
}

const std::shared_ptr<Edge> &Node::getInputEdge() const { return m_inputEdge; }

const std::shared_ptr<Edge> &Node::getOutputEdge() const {
  return m_outputEdge;
}

void Node::connectTo(Node::Ptr targetNode) {
  if (targetNode) {
    targetNode->m_inputEdge = m_outputEdge;
  }
}

void Node::setGraph(Graph *graph) { m_graph = graph; }

Graph *Node::getGraph() const { return m_graph; }

void Node::setShouldCreateNewData(bool should_create) {
  should_create_new_data = should_create;
}

bool Node::getShouldCreateNewData() const { return should_create_new_data; }

} // namespace mainera