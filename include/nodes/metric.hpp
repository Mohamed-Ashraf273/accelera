#ifndef METRIC_NODE_HPP
#define METRIC_NODE_HPP

#include "core/node.hpp"

namespace mainera {

class MAINERA_API MetricNode : public Node {
public:
  MetricNode(const std::string &name, py::object py_func);
  void execute() override;
  void setMetricFlag(bool flag);
  bool getMetricFlag();
  void setInjectedYTrue(py::object y_true);
  py::object getInjectedYTrue() const;

private:
  bool m_enable = true;
  py::object m_injected_y_true;
};
} // namespace mainera
#endif // METRIC_NODE_HPP