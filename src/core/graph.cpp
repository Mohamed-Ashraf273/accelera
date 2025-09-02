#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/branch.hpp"
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace aistudio {

// Constructor/Destructor
Graph::Graph()
    : m_compiled(false), m_parallel_enabled(false), m_first_node(nullptr) {}

Graph::~Graph() { clear(); }

// Core node management
Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func, size_t num_inputs,
                          size_t num_outputs) {
  auto node =
      NodeFactory::createNode(type, name, num_inputs, num_outputs, py_func);
  addNode(node);
  return node;
}

void Graph::addNode(Node::Ptr node) {
  m_nodes.push_back(node);
  m_node_map[node->name] = node; // Fast lookup
  m_compiled = false;

  // Track first node with priority: PREPROCESS > BRANCH > MODEL
  if (!m_first_node || shouldReplaceFirstNode(node)) {
    m_first_node = node;
  }
}

// Utility methods
Node::Ptr Graph::findNodeByName(const std::string &name) const {
  auto it = m_node_map.find(name);
  return (it != m_node_map.end()) ? it->second : nullptr;
}

bool Graph::shouldReplaceFirstNode(Node::Ptr node) const {
  // Only replace if we don't have a first node yet, or if the current first
  // node is not a preprocessing node and the new node is a preprocessing node
  if (node->type == NodeType::PREPROCESS &&
      m_first_node->type != NodeType::PREPROCESS)
    return true;
  if (node->type == NodeType::BRANCH &&
      m_first_node->type != NodeType::PREPROCESS &&
      m_first_node->type != NodeType::BRANCH)
    return true;
  if (node->type == NodeType::MODEL &&
      m_first_node->type != NodeType::PREPROCESS &&
      m_first_node->type != NodeType::BRANCH &&
      m_first_node->type != NodeType::MODEL)
    return true;
  return false;
}

// Graph operations
void Graph::compile() {
  if (m_compiled)
    return;
  m_execution_order = topologicalSort();
  optimizeGraph();
  m_compiled = true;
}

std::vector<py::object> Graph::execute(py::object X, py::object y) {
  if (!m_compiled)
    compile();

  // Provide inputs to first node
  if (m_first_node) {
    // First node always gets input data directly through its input edges
    if (!m_first_node->getInputEdges().empty()) {
      m_first_node->getInputEdges()[0]->setData(X);
      if (m_first_node->getInputEdges().size() > 1 && !y.is_none()) {
        m_first_node->getInputEdges()[1]->setData(y);
      }
    }
  }

  // Execute nodes
  if (m_parallel_enabled) {
    runParallel();
  } else {
    run();
  }

  // Collect predictions
  std::vector<py::object> predictions;
  for (const auto &node : m_nodes) {
    if (node->type == NodeType::PREDICT && !node->getOutputEdges().empty() &&
        node->getOutputEdges()[0]->isReady()) {
      predictions.push_back(node->getOutputEdges()[0]->getData());
    }
  }
  return predictions;
}

void Graph::clear() {
  m_nodes.clear();
  m_execution_order.clear();
  m_node_map.clear();
  m_sequential_nodes.clear();
  m_branch_tails.clear();
  m_compiled = false;
  m_is_branched = false;
  m_first_node = nullptr;
  m_node_counter = 0;
}

// Parallel execution
void Graph::enableParallelExecution(bool enable) {
  m_parallel_enabled = enable;
}

void Graph::runParallel() {
  if (!m_parallel_enabled) {
    run();
    return;
  }

  auto sorted_nodes = topologicalSort();
  std::vector<std::vector<Node::Ptr>> execution_levels;
  std::vector<int> node_levels(sorted_nodes.size(), -1);

  // Calculate execution levels for parallel execution
  for (size_t i = 0; i < sorted_nodes.size(); ++i) {
    int max_input_level = -1;
    auto &node = sorted_nodes[i];

    for (auto &input_edge : node->getInputEdges()) {
      if (input_edge) {
        for (size_t j = 0; j < i; ++j) {
          auto &dep_node = sorted_nodes[j];
          for (auto &output_edge : dep_node->getOutputEdges()) {
            if (output_edge == input_edge) {
              max_input_level = std::max(max_input_level, node_levels[j]);
              break;
            }
          }
        }
      }
    }

    int current_level = max_input_level + 1;
    node_levels[i] = current_level;

    if (current_level >= (int)execution_levels.size()) {
      execution_levels.resize(current_level + 1);
    }
    execution_levels[current_level].push_back(node);
  }

  // Execute levels with parallelism within each level
  for (auto &level : execution_levels) {
    if (level.size() == 1) {
      try {
        py::gil_scoped_acquire acquire;
        level[0]->execute();
        level[0]->dirty = false;
      } catch (const std::exception &e) {
        std::cerr << "Error executing node '" << level[0]->name
                  << "': " << e.what() << std::endl;
        throw;
      }
    } else {
      // Parallel execution
      std::vector<std::thread> threads;
      std::vector<std::exception_ptr> exceptions(level.size());

      for (size_t i = 0; i < level.size(); ++i) {
        threads.emplace_back([&level, &exceptions, i]() {
          try {
            py::gil_scoped_acquire acquire;
            level[i]->execute();
            level[i]->dirty = false;
          } catch (...) {
            exceptions[i] = std::current_exception();
          }
        });
      }

      {
        py::gil_scoped_release release;
        for (auto &thread : threads) {
          thread.join();
        }
      }

      for (size_t i = 0; i < exceptions.size(); ++i) {
        if (exceptions[i]) {
          std::cerr << "Error executing node '" << level[i]->name
                    << "' in parallel" << std::endl;
          std::rethrow_exception(exceptions[i]);
        }
      }
    }
  }
}

void Graph::run() {
  if (!m_compiled)
    compile();

  for (auto &node : m_execution_order) {
    if (node->dirty) {
      try {
        node->execute();
        node->dirty = false;
      } catch (const std::exception &e) {
        std::cerr << "Error executing node '" << node->name << "': " << e.what()
                  << std::endl;
        throw;
      }
    }
  }
}

// Accessors
const std::vector<Node::Ptr> &Graph::getNodes() const { return m_nodes; }

bool Graph::isCompiled() const { return m_compiled; }

// Topological sort implementation
std::vector<Node::Ptr> Graph::topologicalSort() {
  std::vector<Node::Ptr> sorted_nodes;
  std::unordered_set<Node *> visited;
  std::unordered_set<Node *> temp_visited;

  for (auto &node : m_nodes) {
    if (visited.find(node.get()) == visited.end()) {
      topologicalSortDFS(node, visited, temp_visited, sorted_nodes);
    }
  }

  std::reverse(sorted_nodes.begin(), sorted_nodes.end());
  return sorted_nodes;
}

void Graph::topologicalSortDFS(Node::Ptr node,
                               std::unordered_set<Node *> &visited,
                               std::unordered_set<Node *> &temp_visited,
                               std::vector<Node::Ptr> &result) {
  if (temp_visited.find(node.get()) != temp_visited.end()) {
    throw std::runtime_error("Graph contains cycles - not a DAG");
  }

  if (visited.find(node.get()) != visited.end()) {
    return;
  }

  temp_visited.insert(node.get());

  // Visit all dependent nodes (nodes that consume this node's outputs)
  for (const auto &consumer : m_nodes) {
    for (const auto &input_edge : consumer->getInputEdges()) {
      for (const auto &output_edge : node->getOutputEdges()) {
        if (input_edge == output_edge) {
          topologicalSortDFS(consumer, visited, temp_visited, result);
        }
      }
    }
  }

  temp_visited.erase(node.get());
  visited.insert(node.get());
  result.push_back(node);
}

void Graph::optimizeGraph() {
  // Placeholder for future optimizations
  // - Node fusion
  // - Constant propagation
  // - Dead code elimination
}

void Graph::serialize(const std::string &filepath) const {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  // Write OpenVINO-compatible XML format
  file << "<?xml version=\"1.0\"?>\n";
  file << "<net name=\"AI_Studio_Pipeline\" version=\"11\">\n";

  // Add layers (nodes)
  file << "\t<layers>\n";
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];

    // Use the actual node type names
    std::string layer_type;
    switch (node->type) {
    case NodeType::PREPROCESS:
      layer_type = "PREPROCESS";
      break;
    case NodeType::FEATURE:
      layer_type = "FEATURE";
      break;
    case NodeType::MODEL:
      layer_type = "MODEL";
      break;
    case NodeType::PREDICT:
      layer_type = "PREDICT";
      break;
    case NodeType::BRANCH:
      layer_type = "BRANCH";
      break;
    default:
      layer_type = "UNKNOWN";
      break;
    }

    file << "\t\t<layer id=\"" << i << "\" name=\"" << node->name
         << "\" type=\"" << layer_type << "\">\n";

    // Add data section for parameters
    if (!node->py_func.is_none() && py::hasattr(node->py_func, "get_params")) {
      try {
        py::dict params = node->py_func.attr("get_params")();
        file << "\t\t\t<data";
        for (auto item : params) {
          std::string key = py::str(item.first).cast<std::string>();
          std::string value = py::str(item.second).cast<std::string>();
          // Clean up value for XML
          if (value != "None") {
            file << " " << key << "=\"" << value << "\"";
          }
        }
        file << "/>\n";
      } catch (...) {
        file << "\t\t\t<data/>\n";
      }
    } else {
      file << "\t\t\t<data/>\n";
    }

    // Add input ports
    if (node->getInputEdges().size() > 0) {
      file << "\t\t\t<input>\n";
      for (size_t in_idx = 0; in_idx < node->getInputEdges().size(); ++in_idx) {
        file << "\t\t\t\t<port id=\"" << in_idx << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>1</dim>\n";  // Batch size
        file << "\t\t\t\t\t<dim>-1</dim>\n"; // Dynamic dimension
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</input>\n";
    }

    // Add output ports
    if (node->getOutputEdges().size() > 0) {
      file << "\t\t\t<output>\n";
      for (size_t out_idx = 0; out_idx < node->getOutputEdges().size();
           ++out_idx) {
        file << "\t\t\t\t<port id=\""
             << (node->getInputEdges().size() + out_idx)
             << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>1</dim>\n";  // Batch size
        file << "\t\t\t\t\t<dim>-1</dim>\n"; // Dynamic dimension
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</output>\n";
    }

    file << "\t\t</layer>\n";
  }
  file << "\t</layers>\n";

  // Add edges (connections)
  file << "\t<edges>\n";
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];
    const auto &outputs = node->getOutputEdges();

    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
      const auto &edge = outputs[out_idx];
      if (!edge)
        continue;

      // Find which node this edge connects to
      for (size_t j = 0; j < m_nodes.size(); ++j) {
        if (i == j)
          continue; // Skip self
        const auto &target_node = m_nodes[j];
        const auto &inputs = target_node->getInputEdges();

        for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
          if (inputs[in_idx] == edge) {
            // Calculate port IDs according to OpenVINO convention
            size_t from_port = node->getInputEdges().size() + out_idx;
            size_t to_port = in_idx;

            file << "\t\t<edge from-layer=\"" << i << "\" from-port=\""
                 << from_port << "\" to-layer=\"" << j << "\" to-port=\""
                 << to_port << "\"/>\n";
          }
        }
      }
    }
  }
  file << "\t</edges>\n";

  // Add meta information
  file << "\t<meta_data>\n";
  file << "\t\t<MO_version value=\"AI Studio Graph Serializer\"/>\n";
  file << "\t\t<cli_parameters>\n";
  file << "\t\t\t<input_shape value=\"[1,-1]\"/>\n";
  file << "\t\t\t<transform value=\"\"/>\n";
  file << "\t\t</cli_parameters>\n";
  file << "\t</meta_data>\n";

  file << "</net>\n";

  file.close();
}

// Pipeline management methods - optimized with fast lookups
void Graph::addSequentialNode(const std::string &node_type,
                              const std::string &name, py::object obj) {
  // Create node with appropriate input/output configuration
  Node::Ptr node;
  if (node_type == "preprocess") {
    node = add_node(NodeType::PREPROCESS, name, obj, 2, 2);
  } else if (node_type == "model") {
    node = add_node(NodeType::MODEL, name, obj, 2, 1);
  } else if (node_type == "predict") {
    node = add_node(NodeType::PREDICT, name, obj, 2, 1);
  } else {
    throw std::runtime_error("Unknown node type: " + node_type);
  }

  // Connect to previous sequential node if exists
  if (!m_sequential_nodes.empty()) {
    Node::Ptr prev_node = findNodeByName(m_sequential_nodes.back());
    if (prev_node) {
      prev_node->connectTo(0, node, 0);
    }
  }

  m_sequential_nodes.push_back(name);
}

void Graph::startBranching(const std::string &branch_name,
                           const std::vector<py::object> &branch_models) {
  if (m_is_branched) {
    throw std::runtime_error("Already in branched mode - merge first");
  }
  if (m_sequential_nodes.empty()) {
    throw std::runtime_error("Cannot branch - no previous node to branch from");
  }

  Node::Ptr branch_from = findNodeByName(m_sequential_nodes.back());
  if (!branch_from) {
    throw std::runtime_error("Cannot find branch source node: " +
                             m_sequential_nodes.back());
  }

  m_is_branched = true;
  m_branch_tails.clear();

  // Create branches for each model
  for (size_t i = 0; i < branch_models.size(); ++i) {
    py::object branch_model = branch_models[i];

    if (!py::hasattr(branch_model, "fit")) {
      throw std::runtime_error("Branch node must be a model with 'fit' method");
    }

    std::string branch_node_name = branch_name + "_path_" + std::to_string(i) +
                                   "_" + std::to_string(++m_node_counter);
    Node::Ptr branch_node =
        add_node(NodeType::MODEL, branch_node_name, branch_model, 2, 1);

    // Connect both X and y from the branching point
    branch_from->connectTo(0, branch_node, 0);
    branch_from->connectTo(1, branch_node, 1);

    m_branch_tails.push_back(branch_node_name);
  }
}

void Graph::addToBranches(const std::string &node_type, const std::string &name,
                          py::object obj) {
  if (!m_is_branched) {
    throw std::runtime_error("Not in branched mode - cannot add to branches");
  }

  std::vector<std::string> new_branch_tails;

  for (size_t i = 0; i < m_branch_tails.size(); ++i) {
    Node::Ptr tail_node = findNodeByName(m_branch_tails[i]);
    if (!tail_node) {
      throw std::runtime_error("Cannot find branch tail node: " +
                               m_branch_tails[i]);
    }

    std::string branch_node_name = name + "_br" + std::to_string(i) + "_" +
                                   std::to_string(++m_node_counter);

    // Create node based on type
    Node::Ptr branch_node;
    if (node_type == "preprocess") {
      branch_node = add_node(NodeType::PREPROCESS, branch_node_name, obj, 2, 2);
    } else if (node_type == "model") {
      branch_node = add_node(NodeType::MODEL, branch_node_name, obj, 2, 1);
    } else if (node_type == "predict") {
      branch_node = add_node(NodeType::PREDICT, branch_node_name, obj, 2, 1);
    } else {
      throw std::runtime_error("Unknown node type: " + node_type);
    }

    tail_node->connectTo(0, branch_node, 0);
    new_branch_tails.push_back(branch_node_name);
  }

  m_branch_tails = new_branch_tails;
}

void Graph::mergeBranches(const std::string &merge_name,
                          py::object merge_func) {
  if (!m_is_branched) {
    throw std::runtime_error("Not in branched mode - nothing to merge");
  }

  Node::Ptr merge_node = add_node(NodeType::FEATURE, merge_name, merge_func,
                                  m_branch_tails.size(), 1);

  // Connect all branch tails to merge node
  for (size_t i = 0; i < m_branch_tails.size(); ++i) {
    Node::Ptr tail_node = findNodeByName(m_branch_tails[i]);
    if (tail_node) {
      tail_node->connectTo(0, merge_node, i);
    }
  }

  m_is_branched = false;
  m_branch_tails.clear();
  m_sequential_nodes.push_back(merge_name);
}

bool Graph::isBranched() const { return m_is_branched; }

} // namespace aistudio