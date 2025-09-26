#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "nodes/metric.hpp"

#include "nodes/input.hpp"

namespace py = pybind11;

namespace mainera {

MetricNode::MetricNode(const std::string &name, py::object py_func)
    : Node(NodeType::METRIC, name, py_func) {}
void MetricNode::execute() {
  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Metric node '" + name +
                               "' requires a valid input");
    }

    if (!input || input->type != NodeType::PREDICT) {
      throw std::runtime_error("Metric node '" + name +
                               "' requires a prediction input");
    }

    py::object y_pred = input->getData();
    py::object y_true = py_func["y_true"];
    py::object metric_obj = py_func["func"];
    py::object result = metric_obj.attr("execute")(y_true, y_pred);
    setData(result);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error in MetricNode::execute(): " +
                             std::string(e.what()));
  }
}
} // namespace mainera