#include "core/node_factory.hpp"

#include "nodes/feature.hpp"
#include "nodes/merge.hpp"
#include "nodes/metric.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"
#include "nodes/preprocess.hpp"

#include <stdexcept>

namespace mainera {

Node::Ptr NodeFactory::createNode(NodeType type, const std::string &name, py::object py_func) {
    switch (type) {
        case NodeType::PREPROCESS:
            return std::make_shared<PreprocessNode>(name, py_func);
        case NodeType::FEATURE:
            return std::make_shared<FeatureNode>(name, py_func);
        case NodeType::MODEL:
            return std::make_shared<ModelNode>(name, py_func);
        case NodeType::PREDICT:
            return std::make_shared<PredictNode>(name, py_func);
        case NodeType::MERGE:
            return std::make_shared<MergeNode>(name, py_func);
        case NodeType::METRIC:
            return std::make_shared<MetricNode>(name, py_func);
        default:
            throw std::invalid_argument("Unknown node type: "
                                        + std::to_string(static_cast<int>(type)));
    }
}

} // namespace mainera