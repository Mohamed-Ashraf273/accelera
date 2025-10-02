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

private:
  bool m_enable = true;
};
} // namespace mainera
#endif // METRIC_NODE_HPP