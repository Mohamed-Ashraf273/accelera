#include "nodes/input.hpp"

namespace aistudio {

InputNode::InputNode() : Node(NodeType::INPUT, "input_node", 0, 2, py::none()) {
  // Input node has no inputs (data comes from setInputData)
  // and 2 outputs (X and y)
}

void InputNode::execute() {
  if (!m_data_set) {
    throw std::runtime_error("Input data not set before execution");
  }

  // Set the outputs to the input data
  if (!m_outputEdges.empty()) {
    m_outputEdges[0]->setData(m_X); // First output is X

    if (m_outputEdges.size() > 1 && !m_y.is_none()) {
      m_outputEdges[1]->setData(m_y); // Second output is y (if provided)
    }
  }
}

void InputNode::setInputData(py::object X, py::object y) {
  m_X = X;
  m_y = y;
  m_data_set = true;
}

} // namespace aistudio
