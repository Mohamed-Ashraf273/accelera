#include <stdexcept>

#include "core/graph.hpp"
#include "core/node.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/metric.hpp"

namespace mainera {

Node::Node(NodeType type, const std::string &name, py::object py_func)
    : type(type), name(name), py_func(py_func) {}

Node::Ptr Node::clone() const {
  auto new_node =
      NodeFactory::createNode(NodeType(this->type), this->name, this->py_func);

  if (!new_node) {
    throw std::runtime_error("cloneCommonData: new_node is null");
  }

  new_node->setShouldCreateNewData(this->getShouldCreateNewData());
  new_node->setUsesGPU(this->getUsesGPU());

  if (this->type == NodeType::METRIC) {
    auto metric_node = std::dynamic_pointer_cast<MetricNode>(new_node);
    if (metric_node) {
      metric_node->setMetricFlag(false);
    }
    metric_node->py_func = this->py_func;
    return new_node;
  }

  new_node->setData(this->getData());
  return new_node;
}

void Node::setSourceNode(std::shared_ptr<Node> source) {
  m_sourceNode = {source};
}

void Node::setSourceNodes(
    const std::vector<std::shared_ptr<Node>> &source_nodes) {
  m_sourceNode = source_nodes;
}

std::shared_ptr<Node> Node::getSourceNode() const {
  return m_sourceNode.empty() ? nullptr : m_sourceNode[0];
}

const std::vector<std::shared_ptr<Node>> Node::getSourceNodes() const {
  return m_sourceNode;
}

void Node::setData(std::shared_ptr<py::object> result) { m_data = result; }

std::shared_ptr<py::object> Node::getData() const { return m_data; }

void Node::setGraph(Graph *graph) { m_graph = graph; }

Graph *Node::getGraph() const { return m_graph; }

void Node::setShouldCreateNewData(bool should_create) {
  should_create_new_data = should_create;
}

bool Node::getShouldCreateNewData() const { return should_create_new_data; }

void Node::setUsesGPU(bool uses_gpu) { m_uses_gpu = uses_gpu; }

bool Node::getUsesGPU() const { return m_uses_gpu; }

void Node::usesGPU() {
  try {
    py::object model = py_func;

    if (py::hasattr(model, "device")) {
      py::object device_attr = model.attr("device");
      std::string device_str = device_attr.attr("type").cast<std::string>();
      setUsesGPU(device_str == "cuda");
    }

    if (py::hasattr(model, "model")) {
      py::object inner_model = model.attr("model");
      if (py::hasattr(inner_model, "device")) {
        py::object device_attr = inner_model.attr("device");
        std::string device_str = device_attr.attr("type").cast<std::string>();
        setUsesGPU(device_str == "cuda");
      }
    }

  } catch (const py::error_already_set &e) {
    PyErr_Clear();
  }
}

} // namespace mainera