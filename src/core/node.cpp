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

} // namespace mainera