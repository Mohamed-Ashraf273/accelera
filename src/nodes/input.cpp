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
}

void InputNode::setInputData(py::object X, py::object y) {
    m_X = X;
    m_y = y;
    m_data_set = true;
}

} // namespace mainera
