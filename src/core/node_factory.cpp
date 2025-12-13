#include <stdexcept>

#include "core/node_factory.hpp"
#include "nodes/merge.hpp"
#include "nodes/metric.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"
#include "nodes/preprocess.hpp"

namespace accelera {

Node::Ptr NodeFactory::createNode(NodeType type, const std::string &name,
                                  py::object py_func) {
  switch (type) {
  case NodeType::PREPROCESS:
    return std::make_shared<PreprocessNode>(name, py_func);
  case NodeType::MODEL:
    return std::make_shared<ModelNode>(name, py_func);
  case NodeType::PREDICT:
    return std::make_shared<PredictNode>(name, py_func);
  case NodeType::MERGE:
    return std::make_shared<MergeNode>(name, py_func);
  case NodeType::METRIC:
    return std::make_shared<MetricNode>(name, py_func);
  default:
    throw std::invalid_argument("Unknown node type: " +
                                std::to_string(static_cast<int>(type)));
  }
}

Node::Ptr NodeFactory::createNodeCopy(Node::Ptr node, int i) {
  std::string copyName = node->name + "_" + std::to_string(i);
  return NodeFactory::createNode(node->type, copyName, node->py_func);
}

} // namespace accelera