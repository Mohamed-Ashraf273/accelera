#ifndef INPUT_HPP
#define INPUT_HPP

#include "core/node.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

class InputNode : public Node {
public:
  InputNode();
  virtual ~InputNode() = default;

  void execute() override;

  // Set the input data when execution begins
  void setInputData(py::object X, py::object y = py::object());
  void setInputData(py::object X, py::object y, py::object fitted_model);

  // Getter methods for accessing data
  py::object getX() const { return m_X; }
  py::object getY() const { return m_y; }
  py::object getFittedModel() const { return m_fitted_model; }
  void setFittedModel(py::object model) { m_fitted_model = model; }

private:
  py::object m_X;
  py::object m_y;
  py::object m_fitted_model = py::none();
  bool m_data_set = false;
};

} // namespace mainera

#endif // INPUT_HPP
