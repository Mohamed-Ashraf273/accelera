#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <map>
#include <queue>
#include <semaphore>
#include <stack>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/merge.hpp"
#include "nodes/metric.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"
#include "nodes/preprocess.hpp"
#include "utils/graph_utils.hpp"

namespace accelera {

namespace {

bool shouldReleaseData(const Node::Ptr &node) {
  if (!node) {
    return false;
  }
  return node->type != NodeType::MODEL && node->type != NodeType::METRIC;
}

void releaseNodeData(const Node::Ptr &node) {
  if (node->type == NodeType::INPUT) {
    if (node->getGraph() && node->getGraph()->getIsExecuted()) {
      return;
    }

    auto input_node = std::dynamic_pointer_cast<InputNode>(node);
    if (input_node) {
      input_node->clearInputData();
    }
    return;
  }

  node->setData(std::make_shared<py::object>(py::none()));
}

std::unordered_map<Node::Ptr, size_t>
buildRemainingConsumerCounts(const std::vector<Node::Ptr> &nodes) {
  std::unordered_map<Node::Ptr, size_t> consumer_counts;
  for (const auto &node : nodes) {
    consumer_counts[node] = 0;
  }

  for (const auto &node : nodes) {
    for (const auto &source : node->getSourceNodes()) {
      if (source) {
        consumer_counts[source]++;
      }
    }
  }

  return consumer_counts;
}

bool is_sklearn_instance(const py::object &obj) {
  if (obj.is_none()) {
    return false;
  }

  try {
    py::object module = obj.attr("__class__").attr("__module__");
    std::string module_str = py::cast<std::string>(module);
    return module_str.find("sklearn") != std::string::npos;
  } catch (const py::error_already_set &) {
    return false;
  }
}

bool shouldCopyPreprocessInput(const Node::Ptr &node) {
  if (!node || node->type != NodeType::PREPROCESS) {
    return false;
  }

  py::dict params = node->py_func.cast<py::dict>();
  py::object func = py::none();
  if (params.contains("func")) {
    func = params["func"];
  }

  if (!is_sklearn_instance(func)) {
    return true;
  }

  try {
    if (py::hasattr(func, "get_params")) {
      py::dict transformer_params =
          func.attr("get_params")(py::arg("deep") = false);
      if (transformer_params.contains("copy") &&
          !py::cast<bool>(transformer_params["copy"])) {
        return true;
      }
    }
  } catch (const py::error_already_set &) {
    PyErr_Clear();
    return true; // be conservative on inspection failure
  }

  return false;
}

void clearExecutedGraphValidationData(Graph *graph) {
  if (!graph) {
    return;
  }

  for (const auto &node : graph->getNodes()) {
    if (!node) {
      continue;
    }

    if (node->type == NodeType::PREDICT &&
        py::isinstance<py::dict>(node->py_func)) {
      py::dict copied_func = node->py_func.cast<py::dict>().attr("copy")();
      copied_func["test_data"] = py::none();
      node->py_func = copied_func;
      continue;
    }

    if (node->type != NodeType::METRIC) {
      continue;
    }

    try {
      py::object copied_metric =
          py::module_::import("copy").attr("deepcopy")(node->py_func);
      if (py::hasattr(copied_metric, "y_true")) {
        copied_metric.attr("y_true") = py::none();
      }
      if (py::hasattr(copied_metric, "X")) {
        copied_metric.attr("X") = py::none();
      }
      node->py_func = copied_metric;
    } catch (const py::error_already_set &) {
      PyErr_Clear();
    }
  }
}

void clearTrainingGraphRuntimeData(Graph *graph) {
  if (!graph) {
    return;
  }

  for (const auto &node : graph->getNodes()) {
    if (!node) {
      continue;
    }

    if (node->type == NodeType::INPUT) {
      auto input_node = std::dynamic_pointer_cast<InputNode>(node);
      if (input_node) {
        input_node->clearInputData();
      }
    } else {
      node->setData(std::make_shared<py::object>(py::none()));
    }

    if (node->type == NodeType::PREPROCESS &&
        py::isinstance<py::dict>(node->py_func)) {
      py::dict params = node->py_func.cast<py::dict>();
      if (params.contains("_original_func")) {
        py::dict copied_params = params.attr("copy")();
        copied_params["func"] = params["_original_func"];
        copied_params.attr("__delitem__")("_original_func");
        node->py_func = copied_params;
      }
    }
  }
}

std::string nodeSignature(NodeType type, const py::object &node_obj) {
  return std::to_string(static_cast<int>(type)) + ":" +
         py::str(node_obj).cast<std::string>();
}

void releaseConsumedSources(
    const Node::Ptr &node,
    std::unordered_map<Node::Ptr, size_t> &remaining_consumers) {
  for (const auto &source : node->getSourceNodes()) {
    if (!source) {
      continue;
    }

    auto count_it = remaining_consumers.find(source);
    if (count_it == remaining_consumers.end() || count_it->second == 0) {
      continue;
    }

    count_it->second--;
    if (count_it->second == 0 && shouldReleaseData(source)) {
      releaseNodeData(source);
    }
  }
}

} // namespace

Graph::Graph() : m_compiled(false), m_parallel_enabled(false) {
  m_input_node =
      std::make_shared<InputNode>("Input_" + std::to_string(m_nodes.size()));
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  input_as_node->setSourceNode(nullptr);
  m_nodes.push_back(input_as_node);
}

Graph::Graph(const Graph &other) {
  m_compiled = other.m_compiled;
  m_parallel_enabled = other.m_parallel_enabled;
  m_multicore_threshold = other.m_multicore_threshold;
  m_is_branched = other.m_is_branched;
  m_executed = other.m_executed;

  std::unordered_map<Node::Ptr, Node::Ptr> node_mapping;

  m_input_node =
      std::make_shared<InputNode>("Input_" + std::to_string(m_nodes.size()));
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
      auto mapping_it = node_mapping.find(node);
      if (mapping_it != node_mapping.end()) {
        m_execution_order.push_back(mapping_it->second);
      }
    }
  }
}

Graph *Graph::clone() const { return new Graph(*this); }

Graph::~Graph() { clear(); }

Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func) {
  auto node = NodeFactory::createNode(
      type, name + "_" + std::to_string(m_nodes.size()), py_func);

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
    for (auto leaf : leaves) {
      if (!validateNodeConnection(node, leaf)) {
        continue;
      }
    }
    node->setSourceNodes(leaves);
    node->setGraph(this);
    m_nodes.push_back(node);
    break;
  default:
    for (size_t i = 0; i < leaves.size(); ++i) {
      if (!validateNodeConnection(node, leaves[i])) {
        continue;
      }

      Node::Ptr nodeToAdd =
          (i == 0) ? node : NodeFactory::createNodeCopy(node, i);
      nodeToAdd->setShouldCreateNewData(
          is_connected_to_input[i] && nodeToAdd->type == NodeType::PREPROCESS);
      nodeToAdd->setShouldCopyInput(is_connected_to_input[i] &&
                                    shouldCopyPreprocessInput(nodeToAdd));
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
  std::vector<size_t> branch_offsets;
  branch_offsets.reserve(branch_objects.size());
  size_t current_offset = 0;
  for (const auto &branch_object : branch_objects) {
    branch_offsets.push_back(current_offset);
    try {
      current_offset += py::len(py::cast<py::list>(branch_object));
    } catch (const py::cast_error &) {
      current_offset += 1;
    }
  }

  // For each leaf, create parallel branches
  for (size_t leaf_idx = 0; leaf_idx < leaves.size(); ++leaf_idx) {
    Node::Ptr leaf = leaves[leaf_idx];
    std::unordered_map<std::string, Node::Ptr> first_branch_nodes;

    for (size_t branch_idx = 0; branch_idx < branch_objects.size();
         ++branch_idx) {
      Node::Ptr current_source = leaf;

      try {
        py::list node_list = py::cast<py::list>(branch_objects[branch_idx]);

        // It's a list - create a chain of nodes for this branch
        for (size_t list_idx = 0; list_idx < node_list.size(); ++list_idx) {
          size_t node_idx = branch_offsets[branch_idx] + list_idx;
          NodeType nodeType;
          if (node_types[node_idx] == "INPUT") {
            nodeType = NodeType::INPUT;
          } else if (node_types[node_idx] == "PREPROCESS") {
            nodeType = NodeType::PREPROCESS;
          } else if (node_types[node_idx] == "MODEL") {
            nodeType = NodeType::MODEL;
          } else if (node_types[node_idx] == "PREDICT") {
            nodeType = NodeType::PREDICT;
          } else if (node_types[node_idx] == "METRIC") {
            nodeType = NodeType::METRIC;
          } else {
            throw std::runtime_error("Unknown node type: " +
                                     node_types[node_idx]);
          }

          std::string uniqueName =
              node_names[node_idx] + "_" + std::to_string(m_nodes.size());
          py::object node_obj = node_list[list_idx];
          Node::Ptr branchNode =
              NodeFactory::createNode(nodeType, uniqueName, node_obj);

          if (!validateNodeConnection(branchNode, current_source)) {
            continue;
          }

          if (list_idx == 0) {
            std::string signature = nodeSignature(nodeType, node_obj);
            auto existing_node = first_branch_nodes.find(signature);
            if (existing_node != first_branch_nodes.end()) {
              current_source = existing_node->second;
              continue;
            }
            first_branch_nodes[signature] = branchNode;
          }

          branchNode->setShouldCreateNewData(
              list_idx == 0 && branchNode->type == NodeType::PREPROCESS);
          branchNode->setShouldCopyInput(list_idx == 0 &&
                                         shouldCopyPreprocessInput(branchNode));

          m_nodes.push_back(branchNode);
          if (branchNode->type == NodeType::METRIC)
            m_metric_nodes.push_back(
                std::dynamic_pointer_cast<MetricNode>(branchNode));
          branchNode->setSourceNode(current_source);
          branchNode->setGraph(this);
          current_source = branchNode;
        }
      } catch (const py::cast_error &) {
        size_t node_idx = branch_offsets[branch_idx];
        NodeType nodeType;
        if (node_types[node_idx] == "INPUT") {
          nodeType = NodeType::INPUT;
        } else if (node_types[node_idx] == "PREPROCESS") {
          nodeType = NodeType::PREPROCESS;
        } else if (node_types[node_idx] == "MODEL") {
          nodeType = NodeType::MODEL;
        } else if (node_types[node_idx] == "PREDICT") {
          nodeType = NodeType::PREDICT;
        } else if (node_types[node_idx] == "METRIC") {
          nodeType = NodeType::METRIC;
        } else {
          throw std::runtime_error("Unknown node type: " +
                                   node_types[node_idx]);
        }

        std::string uniqueName =
            node_names[node_idx] + "_" + std::to_string(m_nodes.size());

        Node::Ptr branchNode = NodeFactory::createNode(
            nodeType, uniqueName, branch_objects[branch_idx]);

        if (!validateNodeConnection(branchNode, current_source)) {
          continue;
        }

        std::string signature =
            nodeSignature(nodeType, branch_objects[branch_idx]);
        auto existing_node = first_branch_nodes.find(signature);
        if (existing_node != first_branch_nodes.end()) {
          current_source = existing_node->second;
          continue;
        }
        first_branch_nodes[signature] = branchNode;

        branchNode->setShouldCreateNewData(branchNode->type ==
                                           NodeType::PREPROCESS);
        branchNode->setShouldCopyInput(shouldCopyPreprocessInput(branchNode));

        m_nodes.push_back(branchNode);
        if (branchNode->type == NodeType::METRIC)
          m_metric_nodes.push_back(
              std::dynamic_pointer_cast<MetricNode>(branchNode));

        branchNode->setSourceNode(current_source);
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
    selectAllPaths();
  }

  std::vector<py::object> final_result;
  if (!getIsExecuted()) {
    Graph *executed_graph = clone();
    executed_graph->setIsExecuted(true);
    clearExecutedGraphValidationData(executed_graph);
    clearTrainingGraphRuntimeData(this);
    py::object executed_graph_obj = py::cast(executed_graph);
    final_result.push_back(executed_graph_obj);
  }

  final_result.insert(final_result.end(), predictions.begin(),
                      predictions.end());
  return final_result;
}

void Graph::setSelectedPath(const std::string &strategy,
                            py::object custom_strategy) {
  resetSelectedPath();
  if (strategy == "max") {
    selectMaxPath();
  } else if (strategy == "min") {
    selectMinPath();
  } else if (strategy == "custom") {
    selectCustomPath(custom_strategy);
  } else if (strategy == "all") {
    selectAllPaths();
  } else {
    throw std::runtime_error("Unknown selection strategy: " + strategy);
  }
}

void Graph::resetSelectedPath() {
  for (const auto &node : m_nodes) {
    node->selected_in_path = false;
  }
}

void Graph::selectMinPath() { findMaxMinMetricNode(false); }

void Graph::selectMaxPath() { findMaxMinMetricNode(true); }

void Graph::selectCustomPath(py::object custom_strategy) {
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

    setPath(selected_metric_node);

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("Error in custom strategy: " +
                             std::string(e.what()));
  }
}

void Graph::selectAllPaths() {
  for (const auto &node : m_nodes) {
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

  setPath(best_metric_node);
}

void Graph::setPath(std::shared_ptr<MetricNode> best_metric_node) {
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
  py::gil_scoped_release release;

  auto remaining_consumers = buildRemainingConsumerCounts(m_execution_order);

  std::map<Node::Ptr, bool> finished_executing;
  std::map<Node::Ptr, bool> started_executing;

  unsigned int num_cores = std::thread::hardware_concurrency();

  if (num_cores == 0) {
    num_cores = 1;
  }

  unsigned int cpu_threads;

  if (num_cores <= 2) {
    cpu_threads = num_cores;
  } else {
    cpu_threads = num_cores - 1;
  }

  if (const char *env_threads = std::getenv("ACCELERA_NUM_THREADS")) {
    int requested_threads = std::atoi(env_threads);
    if (requested_threads > 0) {
      cpu_threads = static_cast<unsigned int>(requested_threads);
    }
  }

  std::counting_semaphore<1024> available_cpu_threads(cpu_threads);
  std::binary_semaphore available_gpu_thread(1);
  std::binary_semaphore available_nodes(1);

  std::vector<std::exception_ptr> exceptions(m_execution_order.size());
  std::vector<std::future<void>> futures;

  std::mutex data_mutex;
  std::mutex gpu_mutex;

  for (const auto &node : m_execution_order) {
    finished_executing[node] = false;
    started_executing[node] = false;
  }

  std::atomic<int> rem_nodes(static_cast<int>(m_execution_order.size()));

  while (rem_nodes.load() != 0) {
    available_nodes.acquire();

    for (size_t i = 0; i < m_execution_order.size(); ++i) {
      const auto &node = m_execution_order[i];

      if (started_executing[node]) {
        continue;
      }

      bool parents_finished = true;

      if (node->getSourceNode() != nullptr) {
        std::lock_guard<std::mutex> lock(data_mutex);

        for (const auto &source : node->getSourceNodes()) {
          if (!finished_executing[source]) {
            parents_finished = false;
            break;
          }
        }
      }

      if (!parents_finished) {
        continue;
      }

      if (node->getUsesGPU()) {
        available_gpu_thread.acquire();
      } else {
        available_cpu_threads.acquire();
      }

      started_executing[node] = true;
      size_t node_index = i;

      futures.emplace_back(std::async(
          std::launch::async,
          [node, this, &finished_executing, &available_nodes, &data_mutex,
           &available_cpu_threads, &available_gpu_thread, &exceptions,
           node_index, &rem_nodes, &gpu_mutex, &remaining_consumers]() {
            py::gil_scoped_acquire acquire;

            try {
              if (node->getUsesGPU()) {
                std::lock_guard<std::mutex> gpu_lock(gpu_mutex);
                node->execute();
              } else {
                node->execute();
              }
            } catch (...) {
              exceptions[node_index] = std::current_exception();
            }

            {
              std::lock_guard<std::mutex> lock(data_mutex);

              releaseConsumedSources(node, remaining_consumers);
              finished_executing[node] = true;
              rem_nodes.fetch_sub(1);
            }

            if (node->getUsesGPU()) {
              available_gpu_thread.release();
            } else {
              available_cpu_threads.release();
            }

            available_nodes.release();
          }));
    }
  }

  for (auto &f : futures) {
    f.wait();
  }

  for (size_t i = 0; i < exceptions.size(); ++i) {
    if (!exceptions[i]) {
      continue;
    }

    Node::Ptr failed_node = m_execution_order[i];
    std::string node_type = failed_node->getUsesGPU() ? "GPU" : "CPU";

    try {
      std::rethrow_exception(exceptions[i]);
    } catch (const std::exception &e) {
      throw std::runtime_error("Error executing " + node_type + " node '" +
                               failed_node->name +
                               "': " + std::string(e.what()));
    }
  }
}

void Graph::run() {
  if (!m_compiled)
    compile();
  auto remaining_consumers = buildRemainingConsumerCounts(m_execution_order);
  for (auto &node : m_execution_order) {
    try {
      node->execute();
      releaseConsumedSources(node, remaining_consumers);
    } catch (const std::exception &e) {
      throw std::runtime_error("Error executing node '" + node->name +
                               "': " + std::string(e.what()));
    }
  }
}

const std::vector<Node::Ptr> &Graph::getNodes() const { return m_nodes; }

bool Graph::isCompiled() const { return m_compiled; }

// https://www.geeksforgeeks.org/dsa/topological-sorting/
std::vector<Node::Ptr> Graph::topologicalSort() {
  if (m_nodes.empty())
    return {};

  std::unordered_map<Node::Ptr, int> node_to_index;
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    node_to_index[m_nodes[i]] = i;
  }

  std::vector<std::vector<int>> adj(m_nodes.size());
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];

    for (size_t j = 0; j < m_nodes.size(); ++j) {
      const auto &other_node = m_nodes[j];

      if (other_node->getSourceNode() == node) {
        adj[i].push_back(j);
      }

      if (other_node->type == NodeType::MERGE) {
        const auto &source_nodes = other_node->getSourceNodes();
        for (const auto &source : source_nodes) {
          if (source == node) {
            adj[i].push_back(j);
            break;
          }
        }
      }
    }
  }

  std::vector<bool> visited(m_nodes.size(), false);
  std::stack<int> result_stack;

  std::function<void(int)> topologicalSortUtil = [&](int v) {
    visited[v] = true;

    for (int consumer : adj[v]) {
      if (!visited[consumer]) {
        topologicalSortUtil(consumer);
      }
    }

    result_stack.push(v);
  };

  for (int i = 0; i < static_cast<int>(m_nodes.size()); ++i) {
    if (!visited[i]) {
      topologicalSortUtil(i);
    }
  }

  std::vector<Node::Ptr> sorted_nodes;
  while (!result_stack.empty()) {
    sorted_nodes.push_back(m_nodes[result_stack.top()]);
    result_stack.pop();
  }

  if (sorted_nodes.size() != m_nodes.size()) {
    throw std::runtime_error("Graph contains cycles - not a DAG. Processed: " +
                             std::to_string(sorted_nodes.size()) + "/" +
                             std::to_string(m_nodes.size()) + " nodes");
  }

  return sorted_nodes;
}

std::vector<Node::Ptr> Graph::findLeafNodes() const {
  std::vector<Node::Ptr> leaves;

  std::unordered_set<Node::Ptr> source_nodes;

  for (const auto &node : m_nodes) {
    if (node->type == NodeType::MERGE) {
      const auto &sources = node->getSourceNodes();
      source_nodes.insert(sources.begin(), sources.end());
    } else {
      Node::Ptr source = node->getSourceNode();
      if (source) {
        source_nodes.insert(source);
      }
    }
  }

  // Leaves are nodes that are not a source for any other node
  for (const auto &node : m_nodes) {
    if (source_nodes.find(node) == source_nodes.end()) {
      leaves.push_back(node);
    }
  }

  return leaves;
}

void Graph::enableDisableMetrics(py::object y_true, py::object enable) {
  bool enable_metrics = py::cast<bool>(enable);
  if (m_metric_nodes.empty()) {
    if (enable_metrics)
      log_warning("No metric nodes found in the graph to enable metrics.");
    else
      log_warning("No metric nodes found in the graph to disable metrics.");
    return;
  }
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

py::dict Graph::getState() const {
  py::dict state;
  py::list nodes;

  for (const auto &node : m_nodes) {
    py::dict node_state;
    node_state["type"] = static_cast<int>(node->type);
    node_state["name"] = node->name;
    node_state["py_func"] = node->py_func;
    node_state["should_create_new_data"] = node->getShouldCreateNewData();
    node_state["should_copy_input"] = node->getShouldCopyInput();
    node_state["uses_gpu"] = node->getUsesGPU();
    node_state["selected_in_path"] = node->selected_in_path;

    auto data = node->getData();
    node_state["data"] = data ? *data : py::none();

    py::list source_indices;
    for (const auto &source : node->getSourceNodes()) {
      auto source_it = std::find(m_nodes.begin(), m_nodes.end(), source);
      if (source_it == m_nodes.end()) {
        source_indices.append(py::none());
      } else {
        source_indices.append(
            static_cast<int>(std::distance(m_nodes.begin(), source_it)));
      }
    }
    node_state["source_indices"] = source_indices;
    nodes.append(node_state);
  }

  state["nodes"] = nodes;
  state["compiled"] = m_compiled;
  state["executed"] = m_executed;
  state["branched"] = m_is_branched;
  state["parallel_enabled"] = m_parallel_enabled;
  state["multicore_threshold"] = m_multicore_threshold;
  return state;
}

void Graph::setState(py::dict state) {
  clear();

  py::list nodes = state["nodes"];
  std::vector<std::vector<int>> source_indices_by_node;
  source_indices_by_node.reserve(nodes.size());

  for (const auto &item : nodes) {
    py::dict node_state = item.cast<py::dict>();
    NodeType node_type = static_cast<NodeType>(node_state["type"].cast<int>());
    std::string node_name = node_state["name"].cast<std::string>();
    py::object py_func = node_state["py_func"];

    Node::Ptr node;
    if (node_type == NodeType::INPUT) {
      m_input_node = std::make_shared<InputNode>(node_name);
      node = std::static_pointer_cast<Node>(m_input_node);
    } else {
      node = NodeFactory::createNode(node_type, node_name, py_func);
    }

    node->setGraph(this);
    node->setShouldCreateNewData(
        node_state["should_create_new_data"].cast<bool>());
    node->setShouldCopyInput(node_state["should_copy_input"].cast<bool>());
    node->setUsesGPU(node_state["uses_gpu"].cast<bool>());
    node->selected_in_path = node_state["selected_in_path"].cast<bool>();
    node->setData(
        std::make_shared<py::object>(node_state["data"].cast<py::object>()));

    if (node_type == NodeType::METRIC) {
      m_metric_nodes.push_back(std::dynamic_pointer_cast<MetricNode>(node));
    }

    std::vector<int> source_indices;
    py::list serialized_sources = node_state["source_indices"];
    for (const auto &source_item : serialized_sources) {
      if (source_item.is_none()) {
        source_indices.push_back(-1);
      } else {
        source_indices.push_back(source_item.cast<int>());
      }
    }
    source_indices_by_node.push_back(source_indices);
    m_nodes.push_back(node);
  }

  for (size_t node_idx = 0; node_idx < source_indices_by_node.size();
       ++node_idx) {
    std::vector<Node::Ptr> sources;
    for (int source_idx : source_indices_by_node[node_idx]) {
      if (source_idx >= 0 && static_cast<size_t>(source_idx) < m_nodes.size()) {
        sources.push_back(m_nodes[source_idx]);
      }
    }
    if (!sources.empty()) {
      m_nodes[node_idx]->setSourceNodes(sources);
    }
  }

  m_compiled = state["compiled"].cast<bool>();
  m_executed = state["executed"].cast<bool>();
  m_is_branched = state["branched"].cast<bool>();
  m_parallel_enabled = state["parallel_enabled"].cast<bool>();
  m_multicore_threshold = state["multicore_threshold"].cast<size_t>();
  m_compiled = false;
}

} // namespace accelera
