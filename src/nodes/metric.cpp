#include "nodes/metric.hpp"
#include "nodes/input.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace mainera
{

    MetricNode::MetricNode(const std::string &name, py::object py_func) : Node(NodeType::METRIC, name, py_func) {}
    void MetricNode::execute()
    {
        try
        {
            std::shared_ptr<InputNode> input = getInput();
            if (!input)
            {
                throw std::runtime_error("Metric node '" + name + "' requires a valid input");
            }
            std::shared_ptr<Node> sourceInput = getSourceNode();
            if (!sourceInput || sourceInput->type != NodeType::PREDICT)
            {
                throw std::runtime_error("Metric node '" + name + "' requires a prediction input");
            }
            py::object func_instance =
                py::module::import("copy").attr("deepcopy")(py_func);

            py::object y_pred = input->getY();
            py::object X = input->getX();
            py::object model = input->getFittedModel();
            py::object y_true = func_instance["y_true"];
            py::object metric_obj = func_instance["func"];
            py::object result = metric_obj(y_true, y_pred);
            auto new_input = std::make_shared<InputNode>();
            new_input->setInputData(X, y_pred);
            new_input->setFittedModel(model);
            new_input->setMetricResult(result);
            setOutput(new_input);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Error in MetricNode::execute(): " +
                                     std::string(e.what()));
        }
    }
}