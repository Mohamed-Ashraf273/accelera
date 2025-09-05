#ifndef INPUT_HPP
#define INPUT_HPP

#include "core/node.hpp"

namespace mainera {

class InputNode : public Node {
public:
  InputNode();
  virtual ~InputNode() = default;

  void execute() override;

  // Set the input data when execution begins
  void setInputData(py::object X, py::object y = py::object());

private:
  py::object m_X;
  py::object m_y;
  bool m_data_set = false;
};

} // namespace mainera

#endif // INPUT_HPP
