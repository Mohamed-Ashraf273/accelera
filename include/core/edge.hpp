#ifndef EDGE_HPP
#define EDGE_HPP

#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

class __attribute__((visibility("default"))) Edge {
public:
  Edge();
  ~Edge();

  py::object getData();
  void setData(py::object data);

  bool isReady() const;
  void setReady(bool status);

private:
  py::object m_data;
  bool m_isReady;
};

} // namespace mainera

#endif // EDGE_HPP