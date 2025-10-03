#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "core/graph.hpp"
#include "nodes/input.hpp"
#include "nodes/metric.hpp"
#include "nodes/predict.hpp"

namespace py = pybind11;

namespace mainera {

MetricNode::MetricNode(const std::string &name, py::object py_func)
    : Node(NodeType::METRIC, name, py_func) {}

void MetricNode::setMetricFlag(bool flag) { m_enable = flag; }

bool MetricNode::getMetricFlag() { return m_enable; }

void MetricNode::execute() {

  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Metric node '" + name +
                               "' requires a valid input");
    }

    std::shared_ptr<py::object> y_pred = input->getData();

    if (!m_enable) {
      setData(y_pred);
      return;
    }

    if (py::hasattr(py_func, "set_X")) {
      auto predict_node = std::dynamic_pointer_cast<PredictNode>(input);
      py_func.attr("set_X")(predict_node->getPreprocessedData()
                                ? *predict_node->getPreprocessedData()
                                : py::none());
    }

    auto output =
        std::make_shared<py::object>(py_func.attr("execute")(*y_pred));

    setData(output);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error in MetricNode::execute(): " +
                             std::string(e.what()));
  }
}
} // namespace mainera