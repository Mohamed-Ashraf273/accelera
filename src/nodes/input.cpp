#include "nodes/input.hpp"
#include <stdexcept>

namespace mainera {

InputNode::InputNode() : Node(NodeType::INPUT, "input_node", py::none()) {
  // Input node has no inputs (data comes from setInputData)
  // and 1 output in the new single-edge architecture
}

void InputNode::execute() {
  if (!m_data_set) {
    throw std::runtime_error("Input data not set before execution");
  }

  // In the new architecture, we need to create an InputNode object to hold the
  // data For now, just set the output edge as ready
  if (getOutputEdge()) {
    // Create a shared_ptr to this InputNode as the data
    auto this_shared = std::static_pointer_cast<InputNode>(shared_from_this());
    getOutputEdge()->setData(this_shared);
  }
}

void InputNode::setInputData(py::object X, py::object y) {
  m_X = X;
  m_y = y;
  m_data_set = true;
}

} // namespace mainera
