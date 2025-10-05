#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>

#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/merge.hpp"
#include "nodes/metric.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"
#include "nodes/preprocess.hpp"
#include "utils/graph_utils.hpp"

namespace mainera {

Graph::Graph() : m_compiled(false), m_parallel_enabled(false) {
  m_input_node = std::make_shared<InputNode>();
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  input_as_node->setSourceNode(nullptr);
  m_nodes.push_back(input_as_node);
}

Graph::Graph(const Graph &other) {
  m_compiled = other.m_compiled;
  m_parallel_enabled = other.m_parallel_enabled;
  m_multicore_threshold = other.m_multicore_threshold;
  m_is_branched = other.m_is_branched;

  std::unordered_map<Node::Ptr, Node::Ptr> node_mapping;

  m_input_node = std::make_shared<InputNode>();
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  input_as_node->setSourceNode(nullptr);
  input_as_node->setGraph(this);
  m_nodes.push_back(input_as_node);

  Node::Ptr original_input_node = other.m_nodes[0];
  node_mapping[original_input_node] = input_as_node;

  for (const auto &original_node : other.m_nodes) {
    if (original_node->type != NodeType::INPUT &&
        original_node->selected_in_path) {
      Node::Ptr new_node = original_node->clone();
      new_node->setGraph(this);
      node_mapping[original_node] = new_node;
      m_nodes.push_back(new_node);
      if (original_node->type == NodeType::METRIC)
        m_metric_nodes.push_back(
            std::dynamic_pointer_cast<MetricNode>(new_node));
    }
  }

  for (const auto &original_node : other.m_nodes) {
    if (original_node->type != NodeType::INPUT &&
        original_node->selected_in_path) {
      Node::Ptr new_node = node_mapping[original_node];

      // Handle nodes with multiple source nodes
      const std::vector<Node::Ptr> &original_sources =
          original_node->getSourceNodes();
      if (!original_sources.empty()) {
        std::vector<Node::Ptr> new_sources;
        new_sources.reserve(original_sources.size());
        for (const auto &original_source : original_sources) {
          new_sources.push_back(node_mapping[original_source]);
        }
        new_node->setSourceNodes(new_sources);
      } else {
        // Handle nodes with single source node
        Node::Ptr original_source = original_node->getSourceNode();
        if (original_source) {
          Node::Ptr new_source = node_mapping[original_source];
          new_node->setSourceNode(new_source);
        }
      }
    }
  }

  if (other.m_compiled) {
    for (const auto &node : other.m_execution_order) {
      m_execution_order.push_back(node_mapping[node]);
    }
  }
}

Graph *Graph::clone() const { return new Graph(*this); }

Graph::~Graph() { clear(); }

Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func) {
  auto node = NodeFactory::createNode(type, name, py_func);

  addNode(node);
  return node;
}

void Graph::addNode(Node::Ptr node) {
  std::vector<Node::Ptr> leaves = findLeafNodes();

  std::vector<bool> is_connected_to_input;
  is_connected_to_input.reserve(leaves.size());

  for (const auto &leaf : leaves) {
    is_connected_to_input.push_back(leaf->type == NodeType::INPUT);
  }

  switch (node->type) {
  case NodeType::MERGE:
    // Multi-source: Connect to ALL leaves
    for (auto leaf : leaves) {
      if (!validateNodeConnection(node, leaf)) {
        throw std::runtime_error(
            "Invalid connection: Cannot connect node of type " +
            nodeTypeToString(node->type) + " to source node of type " +
            nodeTypeToString(leaf->type));
      }
    }
    node->setSourceNodes(leaves);
    node->setGraph(this);
    m_nodes.push_back(node);
    break;
  default:
    for (size_t i = 0; i < leaves.size(); ++i) {
      if (!validateNodeConnection(node, leaves[i])) {
        throw std::runtime_error(
            "Invalid connection: Cannot connect node of type " +
            nodeTypeToString(node->type) + " to source node of type " +
            nodeTypeToString(leaves[i]->type));
      }
      Node::Ptr nodeToAdd =
          (i == 0) ? node : NodeFactory::createNodeCopy(node, i);
      nodeToAdd->setShouldCreateNewData(is_connected_to_input[i]);
      nodeToAdd->setSourceNode(leaves[i]);
      nodeToAdd->setGraph(this);
      m_nodes.push_back(nodeToAdd);
      if (nodeToAdd->type == NodeType::METRIC)
        m_metric_nodes.push_back(
            std::dynamic_pointer_cast<MetricNode>(nodeToAdd));
    }
    break;
  }
  m_compiled = false;
}

void Graph::split(const std::string &branch_name,
                  const std::vector<py::object> &branch_objects,
                  const std::vector<std::string> &node_types,
                  const std::vector<std::string> &node_names) {

  if (branch_objects.empty()) {
    throw std::runtime_error("At least one branch node must be specified");
  }

  std::vector<Node::Ptr> leaves = findLeafNodes();

  if (leaves.empty()) {
    throw std::runtime_error("No leaf nodes found to split from");
  }

  m_is_branched = true;

  // For each leaf, create parallel branches
  for (size_t leaf_idx = 0; leaf_idx < leaves.size(); ++leaf_idx) {
    Node::Ptr leaf = leaves[leaf_idx];

    for (size_t branch_idx = 0; branch_idx < branch_objects.size();
         ++branch_idx) {
      Node::Ptr current_source = leaf;

      try {
        py::list node_list = py::cast<py::list>(branch_objects[branch_idx]);

        // It's a list - create a chain of nodes for this branch
        for (size_t list_idx = 0; list_idx < node_list.size(); ++list_idx) {
          NodeType nodeType;
          if (node_types[branch_idx + list_idx] == "INPUT") {
            nodeType = NodeType::INPUT;
          } else if (node_types[branch_idx + list_idx] == "PREPROCESS") {
            nodeType = NodeType::PREPROCESS;
          } else if (node_types[branch_idx + list_idx] == "FEATURE") {
            nodeType = NodeType::FEATURE;
          } else if (node_types[branch_idx + list_idx] == "MODEL") {
            nodeType = NodeType::MODEL;
          } else if (node_types[branch_idx + list_idx] == "PREDICT") {
            nodeType = NodeType::PREDICT;
          } else if (node_types[branch_idx + list_idx] == "METRIC") {
            nodeType = NodeType::METRIC;
          } else {
            throw std::runtime_error("Unknown node type: " +
                                     node_types[branch_idx + list_idx]);
          }

          std::string uniqueName;
          if (leaves.size() == 1) {
            uniqueName = node_names[branch_idx + list_idx] + "_chain_" +
                         std::to_string(list_idx);
          } else {
            uniqueName = node_names[branch_idx + list_idx] + "_leaf_" +
                         std::to_string(leaf_idx) + "_chain_" +
                         std::to_string(list_idx);
          }

          py::object node_obj = node_list[list_idx];
          Node::Ptr branchNode =
              NodeFactory::createNode(nodeType, uniqueName, node_obj);

          if (!validateNodeConnection(branchNode, current_source)) {
            throw std::runtime_error(
                "Invalid connection: Cannot connect node of type " +
                nodeTypeToString(branchNode->type) +
                " to source node of type " +
                nodeTypeToString(current_source->type));
          }

          branchNode->setShouldCreateNewData(list_idx == 0);

          m_nodes.push_back(branchNode);
          if (branchNode->type == NodeType::METRIC)
            m_metric_nodes.push_back(
                std::dynamic_pointer_cast<MetricNode>(branchNode));
          branchNode->setSourceNode(current_source);
          branchNode->setGraph(this);
          current_source = branchNode;
        }
      } catch (const py::cast_error &) {
        NodeType nodeType;
        if (node_types[branch_idx] == "INPUT") {
          nodeType = NodeType::INPUT;
        } else if (node_types[branch_idx] == "PREPROCESS") {
          nodeType = NodeType::PREPROCESS;
        } else if (node_types[branch_idx] == "FEATURE") {
          nodeType = NodeType::FEATURE;
        } else if (node_types[branch_idx] == "MODEL") {
          nodeType = NodeType::MODEL;
        } else if (node_types[branch_idx] == "PREDICT") {
          nodeType = NodeType::PREDICT;
        } else if (node_types[branch_idx] == "METRIC") {
          nodeType = NodeType::METRIC;
        } else {
          throw std::runtime_error("Unknown node type: " +
                                   node_types[branch_idx]);
        }

        std::string uniqueName;
        if (leaves.size() == 1) {
          uniqueName = node_names[branch_idx];
        } else {
          uniqueName =
              node_names[branch_idx] + "_leaf_" + std::to_string(leaf_idx);
        }

        Node::Ptr branchNode = NodeFactory::createNode(
            nodeType, uniqueName, branch_objects[branch_idx]);

        if (!validateNodeConnection(branchNode, current_source)) {
          throw std::runtime_error(
              "Invalid connection: Cannot connect node of type " +
              nodeTypeToString(branchNode->type) + " to source node of type " +
              nodeTypeToString(current_source->type));
        }

        branchNode->setShouldCreateNewData(true);

        m_nodes.push_back(branchNode);
        if (branchNode->type == NodeType::METRIC)
          m_metric_nodes.push_back(
              std::dynamic_pointer_cast<MetricNode>(branchNode));

        branchNode->setSourceNode(leaf);
        branchNode->setGraph(this);
        current_source = branchNode;
      }
    }
  }

  m_compiled = false;
}

void Graph::compile() {
  if (m_compiled)
    return;
  m_execution_order = topologicalSort();
  setGPUUsage();
  m_compiled = true;
}

std::vector<py::object> Graph::execute(py::object X, py::object y,
                                       py::object select_strategy,
                                       py::object custom_strategy) {

  std::string use_select_strategy = py::cast<std::string>(select_strategy);

  if (!m_compiled)
    compile();

  if (m_input_node) {
    m_input_node->setInputData(X, y);
  }

  if (m_parallel_enabled && m_execution_order.size() >= m_multicore_threshold) {
    runParallel();
  } else {
    run();
  }

  std::vector<py::object> predictions;
  std::vector<Node::Ptr> final_leaves = findLeafNodes();

  for (const auto &leaf : final_leaves) {
    try {
      if (leaf->type == NodeType::PREPROCESS) {
        std::shared_ptr<py::object> data_ptr = leaf->getData();
        if (data_ptr && !data_ptr->is_none()) {
          auto dict = data_ptr->cast<py::dict>();
          py::object result = dict["X"];
          if (!result.is_none()) {
            predictions.push_back(result);
          }
        }
      } else if (leaf->type == NodeType::INPUT) {
        std::shared_ptr<InputNode> leaf_node =
            std::dynamic_pointer_cast<InputNode>(leaf);
        if (!leaf_node) {
          throw std::runtime_error("Failed to cast to InputNode");
        }
        py::object result =
            py::make_tuple(leaf_node->getX(), leaf_node->getY());
        if (!result.is_none()) {
          predictions.push_back(result);
        }
      } else {
        std::shared_ptr<py::object> result_ptr = leaf->getData();
        if (result_ptr && !result_ptr->is_none()) {
          predictions.push_back(*result_ptr);
        }
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Error collecting results from node '" +
                               leaf->name + "': " + e.what());
    }
  }

  if (m_is_branched) {
    setSelectedPath(use_select_strategy, custom_strategy);
  } else {
    selectAllPaths(*this);
  }

  std::vector<py::object> final_result;
  if (!getIsExecuted()) {
    Graph *executed_graph = clone();
    executed_graph->setIsExecuted(true);
    py::object executed_graph_obj = py::cast(executed_graph);
    final_result.push_back(executed_graph_obj);
  }

  final_result.insert(final_result.end(), predictions.begin(),
                      predictions.end());
  return final_result;
}

void Graph::setSelectedPath(const std::string &strategy,
                            py::object custom_strategy) {
  if (strategy == "max") {
    selectMaxPath(*this);
  } else if (strategy == "min") {
    selectMinPath(*this);
  } else if (strategy == "custom") {
    selectCustomPath(*this, custom_strategy);
  } else if (strategy == "all") {
    selectAllPaths(*this);
  } else {
    throw std::runtime_error("Unknown selection strategy: " + strategy);
  }
}

void Graph::selectMinPath(Graph &graph) { findMaxMinMetricNode(false); }

void Graph::selectMaxPath(Graph &graph) { findMaxMinMetricNode(true); }

void Graph::selectCustomPath(Graph &graph, py::object custom_strategy) {
  if (custom_strategy.is_none()) {
    throw std::runtime_error("Custom strategy function not provided.");
  }

  if (m_metric_nodes.empty()) {
    throw std::runtime_error("No metric nodes found for custom path selection");
  }

  try {
    py::list metric_results = py::list();

    for (const auto &node : m_metric_nodes) {
      if (node) {
        auto data = node->getData();
        if (data && !data->is_none()) {
          py::dict metric_dict = py::cast<py::dict>(*data);
          metric_dict["node name"] = node->name;
          metric_results.append(metric_dict);
        }
      }
    }

    py::object selected_result = custom_strategy(metric_results);
    std::string selected_node_name = py::cast<std::string>(selected_result);

    std::shared_ptr<MetricNode> selected_metric_node = nullptr;

    for (const auto &node : m_metric_nodes) {
      if (node && node->name == selected_node_name) {
        selected_metric_node = node;
        break;
      }
    }

    if (!selected_metric_node) {
      throw std::runtime_error("Selected node '" + selected_node_name +
                               "' not found");
    }

    getPath(selected_metric_node);

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("Error in custom strategy: " +
                             std::string(e.what()));
  }
}

void Graph::selectAllPaths(Graph &graph) {
  for (const auto &node : graph.m_nodes) {
    node->selected_in_path = true;
  }
}

void Graph::findMaxMinMetricNode(bool find_max) {
  double best_metric = find_max ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::infinity();
  std::shared_ptr<MetricNode> best_metric_node = nullptr;

  for (const auto &node : m_metric_nodes) {
    if (node) {
      auto data = node->getData();
      if (data && !data->is_none()) {
        try {
          py::dict metric_dict = py::cast<py::dict>(*data);
          py::object result_obj = metric_dict["result"];

          if (!py::isinstance<py::float_>(result_obj) &&
              !py::isinstance<py::int_>(result_obj)) {
            throw std::runtime_error(
                "Built-in strategies (max/min) only support numeric (double) "
                "metric results. "
                "Node '" +
                node->name + "' has result of type '" +
                std::string(py::str(result_obj.get_type())) + "'. " +
                "Please use 'custom' strategy for non-numeric metric "
                "comparisons.");
          }

          double metric_value = py::cast<double>(result_obj);

          bool is_better = find_max ? (metric_value > best_metric)
                                    : (metric_value < best_metric);

          if (is_better) {
            best_metric = metric_value;
            best_metric_node = node;
          }
        } catch (const py::cast_error &e) {
          throw std::runtime_error("Built-in strategies (max/min) only support "
                                   "numeric (double) metric results. "
                                   "Failed to cast result from node '" +
                                   node->name + "' to double. " +
                                   "Please use 'custom' strategy for "
                                   "non-numeric metric comparisons. "
                                   "Cast error: " +
                                   std::string(e.what()));
        } catch (const std::exception &e) {
          throw std::runtime_error("Error processing metric data for node '" +
                                   node->name + "': " + e.what());
        }
      }
    } else {
      throw std::runtime_error("Failed to cast to MetricNode while selecting " +
                               std::string(find_max ? "max" : "min") +
                               " path for node '" + node->name + "'");
    }
  }

  if (!best_metric_node) {
    throw std::runtime_error("No valid metric nodes found for " +
                             std::string(find_max ? "max" : "min") +
                             " path selection");
  }

  getPath(best_metric_node);
}

void Graph::getPath(std::shared_ptr<MetricNode> best_metric_node) {
  Node::Ptr current_node = best_metric_node;
  while (current_node) {
    current_node->selected_in_path = true;
    current_node = current_node->getSourceNode();
  }
}

void Graph::clear() {
  m_nodes.clear();
  m_metric_nodes.clear();
  m_execution_order.clear();
  m_compiled = false;
  m_is_branched = false;
  m_input_node = nullptr;
}

void Graph::enableParallelExecution(bool enable) {
  m_parallel_enabled = enable;
}

void Graph::setMulticoreThreshold(size_t threshold) {
  m_multicore_threshold = threshold;
}

void Graph::setGPUUsage() {
  for (const auto &node : m_nodes) {
    if (!node->py_func.is_none()) {
      node->usesGPU();
    }
  }
}

void Graph::runParallel() {
  auto execution_levels = groupNodesByLevel();

  for (const auto &level : execution_levels) {
    if (level.size() == 1) {
      auto &node = level[0];
      try {
        node->execute();
      } catch (const std::exception &e) {
        throw std::runtime_error("Error executing node '" + node->name +
                                 "': " + std::string(e.what()));
      }
    } else {
      // Release GIL for the main thread so worker threads can acquire it
      py::gil_scoped_release release;
      executeNodesInParallel(level);
    }
  }
}

void Graph::run() {
  if (!m_compiled)
    compile();
  for (auto &node : m_execution_order) {
    try {
      node->execute();
    } catch (const std::exception &e) {
      throw std::runtime_error("Error executing node '" + node->name +
                               "': " + std::string(e.what()));
    }
  }
}

const std::vector<Node::Ptr> &Graph::getNodes() const { return m_nodes; }

bool Graph::isCompiled() const { return m_compiled; }

std::vector<py::object> Graph::getPreprocessingFunctions(Node::Ptr node) const {
  std::vector<py::object> m_preprocessing_functions;

  while (node) {
    py::object py_func = node->py_func;
    if (node->type == NodeType::PREPROCESS && !py_func.is_none()) {
      m_preprocessing_functions.insert(m_preprocessing_functions.begin(),
                                       py_func);
    }
    node = node->getSourceNode();
  }

  return m_preprocessing_functions;
}

std::vector<Node::Ptr> Graph::topologicalSort() {
  if (m_nodes.empty())
    return {};

  // Build adjacency list: node -> list of consumers (nodes that have this
  // node as source)
  std::unordered_map<Node *, std::vector<Node::Ptr>> adjacency_list;
  std::unordered_map<Node *, int> in_degrees;

  for (const auto &node : m_nodes) {
    adjacency_list[node.get()] = {};
    in_degrees[node.get()] = 0;
  }

  // Build connections based on source node relationships
  for (const auto &node : m_nodes) {
    if (auto source = node->getSourceNode()) {
      // node depends on source
      adjacency_list[source.get()].push_back(node);
      in_degrees[node.get()]++;
    }
  }

  // Kahn's algorithm (BFS-based) - O(V + E)
  std::queue<Node::Ptr> queue;
  std::vector<Node::Ptr> sorted_nodes;

  // Start with nodes that have no dependencies (no source node)
  for (const auto &node : m_nodes) {
    if (in_degrees[node.get()] == 0) {
      queue.push(node);
    }
  }

  while (!queue.empty()) {
    Node::Ptr current = queue.front();
    queue.pop();
    sorted_nodes.push_back(current);

    // Process all nodes that depend on current node
    for (const auto &consumer : adjacency_list[current.get()]) {
      in_degrees[consumer.get()]--;
      if (in_degrees[consumer.get()] == 0) {
        queue.push(consumer);
      }
    }
  }

  // Check for cycles
  if (sorted_nodes.size() != m_nodes.size()) {
    throw std::runtime_error("Graph contains cycles - not a DAG. Processed: " +
                             std::to_string(sorted_nodes.size()) + "/" +
                             std::to_string(m_nodes.size()) + " nodes");
  }

  return sorted_nodes;
}

std::vector<Node::Ptr> Graph::findLeafNodes() const {
  std::vector<Node::Ptr> leaves;

  // Create a set of nodes that serve as sources for other nodes
  std::unordered_set<Node::Ptr> source_nodes;

  for (const auto &node : m_nodes) {
    for (auto source : node->getSourceNodes()) {
      source_nodes.insert(source);
    }
  }

  // Leaf nodes are those that are not source nodes for any other node
  for (const auto &node : m_nodes) {
    if (source_nodes.find(node) == source_nodes.end()) {
      leaves.push_back(node);
    }
  }

  return leaves;
}

std::vector<std::vector<Node::Ptr>> Graph::groupNodesByLevel() const {
  if (m_nodes.empty())
    throw std::runtime_error("No nodes in graph");

  std::vector<std::vector<Node::Ptr>> levels;
  std::unordered_map<Node *, int> levels_map;
  std::queue<Node::Ptr> queue;

  // Find root nodes (nodes with no source)
  for (const auto &node : m_nodes) {
    if (!node->getSourceNode()) {
      levels_map[node.get()] = 0;
      queue.push(node);
    }
  }

  // BFS to assign levels
  while (!queue.empty()) {
    Node::Ptr current = queue.front();
    queue.pop();
    int current_level = levels_map[current.get()];

    // Ensure we have enough levels
    if (current_level >= levels.size()) {
      levels.resize(current_level + 1);
    }
    levels[current_level].push_back(current);

    // Find all nodes that have this node as source
    for (const auto &node : m_nodes) {
      if (node->getSourceNode() == current) {
        if (levels_map.find(node.get()) == levels_map.end()) {
          levels_map[node.get()] = current_level + 1;
          queue.push(node);
        }
      }
    }
  }

  return levels;
}

void Graph::executeNodesInParallel(const std::vector<Node::Ptr> &nodes) {
  std::vector<Node::Ptr> cpu_nodes;
  std::vector<Node::Ptr> gpu_nodes;

  for (const auto &node : nodes) {
    if (node->getUsesGPU()) {
      gpu_nodes.push_back(node);
    } else {
      cpu_nodes.push_back(node);
    }
  }

  std::vector<std::exception_ptr> exceptions(nodes.size());
  std::vector<std::future<void>> futures;

  if (!cpu_nodes.empty()) {
    for (size_t i = 0; i < cpu_nodes.size(); ++i) {
      const auto &node = cpu_nodes[i];
      futures.emplace_back(
          std::async(std::launch::async, [node, &exceptions, i]() {
            py::gil_scoped_acquire acquire;
            try {
              node->execute();
            } catch (...) {
              exceptions[i] = std::current_exception();
            }
          }));
    }
  }

  // Launch GPU nodes sequentially BUT CONCURRENTLY with CPU nodes
  if (!gpu_nodes.empty()) {
    futures.emplace_back(
        std::async(std::launch::async,
                   [gpu_nodes, &exceptions, cpu_count = cpu_nodes.size()]() {
                     py::gil_scoped_acquire acquire;
                     try {
                       for (size_t i = 0; i < gpu_nodes.size(); ++i) {
                         gpu_nodes[i]->execute();
                       }
                     } catch (...) {
                       // For GPU nodes, we need to handle exceptions
                       // differently since we're processing multiple nodes in
                       // one thread Let's store the exception for the first
                       // failed GPU node
                       exceptions[cpu_count] = std::current_exception();
                     }
                   }));
  }

  for (auto &future : futures) {
    future.wait();
  }

  for (size_t i = 0; i < exceptions.size(); ++i) {
    if (exceptions[i]) {
      Node::Ptr failed_node;
      std::string node_type;

      if (i < cpu_nodes.size()) {
        failed_node = cpu_nodes[i];
        node_type = "CPU";
      } else if (i == cpu_nodes.size() && !gpu_nodes.empty()) {
        // For GPU nodes, we only have one exception slot for the entire
        // sequence
        failed_node = gpu_nodes[0];
        node_type = "GPU";
      } else {
        failed_node = nodes[i];
        node_type = "unknown";
      }

      try {
        std::rethrow_exception(exceptions[i]);
      } catch (const std::exception &e) {
        throw std::runtime_error("Error executing " + node_type + " node '" +
                                 failed_node->name +
                                 "': " + std::string(e.what()));
      }
    }
  }
}

void Graph::enableDisableMetrics(py::object y_true, py::object enable) {
  bool enable_metrics = py::cast<bool>(enable);
  for (const auto &node : m_metric_nodes) {
    if (node) {
      node->setMetricFlag(enable_metrics);
      if (enable_metrics && !y_true.is_none()) {
        if (!py::hasattr(node->py_func, "set_y_true")) {
          throw std::runtime_error("UnSupervised Metric function " +
                                   node->name + " has no 'set_y_true' method");
        }
        node->py_func.attr("set_y_true")(y_true);
      } else {
        node->setData(std::make_shared<py::object>(py::none()));
      }
    }
  }
}

} // namespace mainera