#include "core/node.hpp"
#include "core/graph.hpp"
#include <iostream>
#include <stdexcept>

namespace aistudio {

Node::Node(NodeType type, const std::string &name, size_t numInputs,
           size_t numOutputs, py::object py_func)
    : type(type), name(name), py_func(py_func) {

  // Initialize input edges
  m_inputEdges.resize(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    m_inputEdges[i] = std::make_shared<Edge>();
  }

  // Initialize output edges
  m_outputEdges.resize(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    m_outputEdges[i] = std::make_shared<Edge>();
  }
}

Node::~Node() {
  // Cleanup handled by shared_ptr
}

py::list Node::collectInputs() {
  py::list inputs;
  for (const auto &edge : m_inputEdges) {
    if (edge && edge->isReady()) {
      inputs.append(edge->getData());
    } else {
      inputs.append(py::none());
    }
  }
  return inputs;
}

void Node::setOutputs(py::object result) {
  if (m_outputEdges.size() == 1) {
    // Single output
    m_outputEdges[0]->setData(result);
  } else if (py::isinstance<py::tuple>(result)) {
    // Multiple outputs as tuple
    py::tuple result_tuple = result.cast<py::tuple>();
    for (size_t i = 0; i < m_outputEdges.size() && i < result_tuple.size();
         ++i) {
      m_outputEdges[i]->setData(result_tuple[i]);
    }
  } else {
    // Single output to first edge only
    m_outputEdges[0]->setData(result);
  }
}

const std::vector<std::shared_ptr<Edge>> &Node::getInputEdges() const {
  return m_inputEdges;
}

const std::vector<std::shared_ptr<Edge>> &Node::getOutputEdges() const {
  return m_outputEdges;
}

void Node::connectTo(size_t myOutputIndex, Node::Ptr targetNode,
                     size_t targetInputIndex) {
  if (myOutputIndex >= m_outputEdges.size()) {
    throw std::out_of_range("Output index out of range for node: " + name);
  }
  if (targetInputIndex >= targetNode->m_inputEdges.size()) {
    throw std::out_of_range("Input index out of range for target node: " +
                            targetNode->name);
  }

  // Share the edge between output and input
  targetNode->m_inputEdges[targetInputIndex] = m_outputEdges[myOutputIndex];
}

void Node::setGraph(Graph *graph) { m_graph = graph; }

Graph *Node::getGraph() const { return m_graph; }

} // namespace aistudio