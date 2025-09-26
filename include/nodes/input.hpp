#ifndef INPUT_HPP
#define INPUT_HPP

#include <pybind11/pybind11.h>

#include "core/node.hpp"

namespace py = pybind11;

namespace mainera {

class MAINERA_API InputNode : public Node {
public:
  InputNode();
  virtual ~InputNode() = default;
  void execute() override;
  void setInputData(py::object X, py::object y = py::object());

  py::object getX() const { return m_X; }
  py::object getY() const { return m_y; }

private:
  py::object m_X;
  py::object m_y;
  bool m_data_set = false;
};

} // namespace mainera

#endif // INPUT_HPP
