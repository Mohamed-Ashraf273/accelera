#include "nodes/metric.hpp"
#include "core/graph.hpp"
#include "nodes/input.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mainera {

MetricNode::MetricNode(const std::string &name, py::object py_func)
    : Node(NodeType::METRIC, name, py_func) {}

void MetricNode::setMetricFlag(bool flag) { m_enable = flag; }

bool MetricNode::getMetricFlag() { return m_enable; }

void MetricNode::setInjectedYTrue(py::object y_true) {
  m_injected_y_true = y_true;
}

py::object MetricNode::getInjectedYTrue() const { return m_injected_y_true; }

void MetricNode::execute() {

  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Metric node '" + name +
                               "' requires a valid input");
    }

    py::object y_pred = input->getData();

    if (!m_enable) {
      setData(y_pred);
      return;
    }

    py::object y_true;
    py::object metric_obj;
    py::object metric_name;
    if (!getGraph()->getIsExecuted()) {
      y_true = py_func["y_true"];

    } else {
      y_true = m_injected_y_true;
    }
    metric_obj = py_func["func"];
    metric_name = py_func["metric_name"];

    py::dict output;
    output["metric_name"] = metric_name;
    output["result"] = metric_obj.attr("execute")(y_true, y_pred);

    setData(output);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error in MetricNode::execute(): " +
                             std::string(e.what()));
  }
}
} // namespace mainera