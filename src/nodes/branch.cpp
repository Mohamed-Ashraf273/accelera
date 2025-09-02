#include "nodes/branch.hpp"
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace aistudio {

BranchNode::BranchNode(const std::string &name, size_t numInputs,
                       size_t numOutputs, py::object py_func)
    : Node(NodeType::BRANCH, name, numInputs, numOutputs, py_func) {}

void BranchNode::execute() {
  // Check if we have the required inputs (X and y)
  if (m_inputEdges.size() < 2 || !m_inputEdges[0] ||
      !m_inputEdges[0]->isReady() || !m_inputEdges[1] ||
      !m_inputEdges[1]->isReady()) {
    return;
  }

  py::object X_data = m_inputEdges[0]->getData();
  py::object y_data = m_inputEdges[1]->getData();

  // Send (X, y) pairs to each model
  // Outputs are arranged as: X1, y1, X2, y2, X3, y3, ...
  size_t num_models = m_outputEdges.size() / 2;

  for (size_t i = 0; i < num_models; i++) {
    size_t x_idx = i * 2;
    size_t y_idx = i * 2 + 1;

    if (x_idx < m_outputEdges.size() && m_outputEdges[x_idx]) {
      m_outputEdges[x_idx]->setData(X_data);
    }

    if (y_idx < m_outputEdges.size() && m_outputEdges[y_idx]) {
      m_outputEdges[y_idx]->setData(y_data);
    }
  }
}

} // namespace aistudio