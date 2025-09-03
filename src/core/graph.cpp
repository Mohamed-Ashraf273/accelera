#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace aistudio {

// Constructor/Destructor
Graph::Graph()
    : m_compiled(false), m_parallel_enabled(false), m_first_node(nullptr) {
  // Automatically create an input node as the starting point
  m_input_node = std::make_shared<InputNode>();
  m_nodes.push_back(m_input_node);
  m_node_map[m_input_node->name] = m_input_node;
  m_first_node = m_input_node;
}

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
  // Find all current leaf nodes before adding the new node
  std::vector<Node::Ptr> leaves = findLeafNodes();

  if (leaves.empty()) {
    // If no leaves exist (empty graph), just add the node normally
    m_nodes.push_back(node);
    m_node_map[node->name] = node; // Fast lookup
    m_compiled = false;

    // Track first node with priority: INPUT > PREPROCESS > MODEL
    if (!m_first_node || shouldReplaceFirstNode(node)) {
      m_first_node = node;
    }
  } else {
    // Create a copy of the node for each leaf and connect them
    for (size_t i = 0; i < leaves.size(); ++i) {
      Node::Ptr nodeToAdd;

      if (i == 0) {
        // Use the original node for the first leaf
        nodeToAdd = node;
      } else {
        // Create a copy for subsequent leaves
        std::string copyName = node->name + "_copy_" + std::to_string(i);
        nodeToAdd = NodeFactory::createNode(
            node->type, copyName, node->getInputEdges().size(),
            node->getOutputEdges().size(), node->py_func);
      }

      // Add the node to the graph
      m_nodes.push_back(nodeToAdd);
      m_node_map[nodeToAdd->name] = nodeToAdd; // Fast lookup

      // Connect the leaf to this node
      // Assuming leaf has at least one output and nodeToAdd has at least one
      // input
      if (!leaves[i]->getOutputEdges().empty() &&
          !nodeToAdd->getInputEdges().empty()) {
        leaves[i]->connectTo(0, nodeToAdd, 0);
      }
    }

    m_compiled = false;

    // Track first node with priority: INPUT > PREPROCESS > MODEL
    if (!m_first_node || shouldReplaceFirstNode(node)) {
      m_first_node = node;
    }
  }
}

void Graph::split(const std::string &branch_name,
                  const std::vector<py::object> &branch_objects,
                  const std::vector<std::string> &node_types,
                  const std::vector<std::string> &node_names) {
  // Validate input parameters
  if (branch_objects.size() != node_types.size() ||
      branch_objects.size() != node_names.size()) {
    throw std::runtime_error(
        "branch_objects, node_types, and node_names must have the same size");
  }

  if (branch_objects.empty()) {
    throw std::runtime_error("At least one branch node must be specified");
  }

  // Get all current leaf nodes
  std::vector<Node::Ptr> leaves = findLeafNodes();

  if (leaves.empty()) {
    throw std::runtime_error("No leaf nodes found to split from");
  }

  // Clear any previous branch state
  m_branch_tails.clear();
  m_is_branched = true;

  // For each leaf, create all the specified nodes and connect them
  for (size_t leaf_idx = 0; leaf_idx < leaves.size(); ++leaf_idx) {
    Node::Ptr leaf = leaves[leaf_idx];

    for (size_t node_idx = 0; node_idx < branch_objects.size(); ++node_idx) {
      // Convert node type string to NodeType enum
      NodeType nodeType;
      if (node_types[node_idx] == "INPUT") {
        nodeType = NodeType::INPUT;
      } else if (node_types[node_idx] == "PREPROCESS") {
        nodeType = NodeType::PREPROCESS;
      } else if (node_types[node_idx] == "FEATURE") {
        nodeType = NodeType::FEATURE;
      } else if (node_types[node_idx] == "MODEL") {
        nodeType = NodeType::MODEL;
      } else if (node_types[node_idx] == "PREDICT") {
        nodeType = NodeType::PREDICT;
      } else {
        throw std::runtime_error("Unknown node type: " + node_types[node_idx]);
      }

      // Create unique name for this branch node
      std::string uniqueName;
      if (leaves.size() == 1) {
        // If only one leaf, use the original name
        uniqueName = node_names[node_idx];
      } else {
        // If multiple leaves, append leaf index to make unique
        uniqueName = node_names[node_idx] + "_leaf_" + std::to_string(leaf_idx);
      }

      // Create the new node
      Node::Ptr branchNode = NodeFactory::createNode(nodeType, uniqueName, 1, 1,
                                                     branch_objects[node_idx]);

      // Add to graph
      m_nodes.push_back(branchNode);
      m_node_map[branchNode->name] = branchNode;

      // Connect the leaf to this branch node
      if (!leaf->getOutputEdges().empty() &&
          !branchNode->getInputEdges().empty()) {
        leaf->connectTo(0, branchNode, 0);
      }

      // Track this as a branch tail for potential future merging
      m_branch_tails.push_back(branchNode->name);
    }
  }

  // Mark graph as needing recompilation
  m_compiled = false;
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

  // Set input data on the input node
  if (m_input_node) {
    m_input_node->setInputData(X, y);
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
  m_input_node = nullptr;
  m_node_counter = 0;

  // Recreate the input node
  m_input_node = std::make_shared<InputNode>();
  m_nodes.push_back(m_input_node);
  m_node_map[m_input_node->name] = m_input_node;
  m_first_node = m_input_node;
}

// Parallel execution
void Graph::enableParallelExecution(bool enable) {
  m_parallel_enabled = enable;
}

void Graph::setMulticoreThreshold(size_t threshold) {
  m_multicore_threshold = threshold;
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

  // Execute levels with optimal parallelism
  for (auto &level : execution_levels) {
    if (level.size() == 1) {
      // Single node - execute directly
      try {
        py::gil_scoped_acquire acquire;
        level[0]->execute();
        level[0]->dirty = false;
      } catch (const std::exception &e) {
        std::cerr << "Error executing node '" << level[0]->name
                  << "': " << e.what() << std::endl;
        throw;
      }
    } else if (level.size() >= m_multicore_threshold) {
      // Use std::async for true multicore execution when beneficial
      std::vector<std::future<void>> futures;
      std::vector<std::exception_ptr> exceptions(level.size());

      for (size_t i = 0; i < level.size(); ++i) {
        futures.emplace_back(
            std::async(std::launch::async, [&level, &exceptions, i]() {
              try {
                py::gil_scoped_acquire acquire;
                level[i]->execute();
                level[i]->dirty = false;
              } catch (...) {
                exceptions[i] = std::current_exception();
              }
            }));
      }

      // Wait for all tasks to complete
      for (auto &future : futures) {
        future.wait();
      }

      // Check for exceptions
      for (size_t i = 0; i < exceptions.size(); ++i) {
        if (exceptions[i]) {
          std::cerr << "Error executing node '" << level[i]->name
                    << "' in parallel" << std::endl;
          std::rethrow_exception(exceptions[i]);
        }
      }
    } else {
      // Use threads for smaller groups to avoid overhead
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

std::vector<Node::Ptr> Graph::findLeafNodes() const {
  std::vector<Node::Ptr> leaves;

  for (const auto &node : m_nodes) {
    bool isLeaf = true;

    // Check if this node has any output edges that are connected to other nodes
    for (const auto &output_edge : node->getOutputEdges()) {
      if (output_edge) {
        // Check if any other node uses this output edge as input
        for (const auto &other_node : m_nodes) {
          if (other_node != node) {
            for (const auto &input_edge : other_node->getInputEdges()) {
              if (input_edge == output_edge) {
                isLeaf = false;
                break;
              }
            }
            if (!isLeaf)
              break;
          }
        }
        if (!isLeaf)
          break;
      }
    }

    if (isLeaf) {
      leaves.push_back(node);
    }
  }

  return leaves;
}

bool Graph::shouldReplaceFirstNode(Node::Ptr node) const {
  if (!m_first_node)
    return true;

  // Priority: INPUT > PREPROCESS > MODEL > others
  if (node->type == NodeType::INPUT)
    return true;
  if (m_first_node->type == NodeType::INPUT)
    return false;

  if (node->type == NodeType::PREPROCESS)
    return true;
  if (m_first_node->type == NodeType::PREPROCESS)
    return false;

  if (node->type == NodeType::MODEL &&
      m_first_node->type != NodeType::PREPROCESS)
    return true;

  return false;
}

Node::Ptr Graph::findNodeByName(const std::string &name) const {
  auto it = m_node_map.find(name);
  return (it != m_node_map.end()) ? it->second : nullptr;
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

} // namespace aistudio