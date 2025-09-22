#ifndef METRIC_NODE_HPP
#define METRIC_NODE_HPP
#include "core/node.hpp"

namespace mainera {

class MAINERA_API MetricNode : public Node {
    public:
    MetricNode(const std::string &name, py::object py_func);
    void execute() override;
};
} // namespace mainera
#endif // METRIC_NODE_HPP