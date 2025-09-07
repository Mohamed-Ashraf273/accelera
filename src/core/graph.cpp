#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/merge.hpp"
#include "passes/node_fusion.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace mainera {

Graph::Graph()
    : m_compiled(false), m_parallel_enabled(false), m_first_node(nullptr) {
  // Automatically create an input node as the starting point
  m_input_node = std::make_shared<InputNode>();
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  m_nodes.push_back(input_as_node);
  m_node_map[m_input_node->name] = input_as_node;
  m_first_node = input_as_node;
}

Graph::~Graph() { clear(); }

Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func) {
  auto node = NodeFactory::createNode(type, name, py_func);

  // Store preprocessing functions for later use in predict nodes
  if (type == NodeType::PREPROCESS && !py_func.is_none()) {
    m_preprocessing_functions.push_back(py_func);
  }

  addNode(node);
  return node;
}

void Graph::addNode(Node::Ptr node) {
  // Set graph pointer for the node
  node->setGraph(this);

  // Find all current leaf nodes before adding the new node
  std::vector<Node::Ptr> leaves = findLeafNodes();

  if (leaves.empty()) {
    // If no leaves exist (empty graph), just add the node normally
    m_nodes.push_back(node);
    m_node_map[node->name] = node;
    m_compiled = false;

    // Track first node with priority: INPUT > PREPROCESS > MODEL
    if (!m_first_node || shouldReplaceFirstNode(node)) {
      m_first_node = node;
    }
  } else {
    // Create a copy of the node for each leaf and connect them
    std::vector<std::string>
        new_branch_tails; // Track new tails if in branched mode

    for (size_t i = 0; i < leaves.size(); ++i) {
      Node::Ptr nodeToAdd;

      if (i == 0) {
        // Use the original node for the first leaf
        nodeToAdd = node;
      } else {
        // Create a copy for subsequent leaves
        std::string copyName = node->name + "_copy_" + std::to_string(i);
        nodeToAdd =
            NodeFactory::createNode(node->type, copyName, node->py_func);
        // Set graph pointer for copied node
        nodeToAdd->setGraph(this);
      }

      // Add the node to the graph
      m_nodes.push_back(nodeToAdd);
      m_node_map[nodeToAdd->name] = nodeToAdd; // Fast lookup

      // Check if this is the first node after input (should create new data)
      bool is_first_after_input = false;
      for (const auto &leaf : leaves) {
        if (leaf->type == NodeType::INPUT) {
          is_first_after_input = true;
          break;
        }
      }

      // Only mark for creating new data if it's first after input
      // Nodes created via split() already have the flag set appropriately
      if (is_first_after_input) {
        nodeToAdd->setShouldCreateNewData(true);
      } else {
        // Default to false for memory optimization - enable InputNode reuse
        nodeToAdd->setShouldCreateNewData(false);
      }

      // Connect the leaf to this node
      if (leaves[i]->getOutputEdge() && nodeToAdd->getInputEdge()) {
        leaves[i]->connectTo(nodeToAdd);
      }

      // If in branched mode, track the new node as a potential branch tail
      if (m_is_branched) {
        new_branch_tails.push_back(nodeToAdd->name);
      }
    }

    // Update branch tails if we're in branched mode
    if (m_is_branched && !new_branch_tails.empty()) {
      m_branch_tails = new_branch_tails;
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

      Node::Ptr branchNode = NodeFactory::createNode(nodeType, uniqueName,
                                                     branch_objects[node_idx]);

      // Mark this as a branch head node - should create new data for memory
      // optimization
      branchNode->setShouldCreateNewData(true);

      // Add to graph
      m_nodes.push_back(branchNode);
      m_node_map[branchNode->name] = branchNode;

      // Connect the leaf to this branch node
      if (leaf->getOutputEdge() && branchNode->getInputEdge()) {
        leaf->connectTo(branchNode);
      }

      // Track this as a branch tail for potential future merging
      m_branch_tails.push_back(branchNode->name);
    }
  }

  // Mark graph as needing recompilation
  m_compiled = false;
}

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

  if (m_input_node) {
    m_input_node->setInputData(X, y);
  }

  if (m_parallel_enabled) {
    runParallel();
  } else {
    run();
  }

  // Collect results from nodes that have computed results
  std::vector<py::object> predictions;

  // First, try to get results from MERGE leaf nodes
  std::vector<Node::Ptr> final_leaves = findLeafNodes();

  bool found_merge_result = false;
  for (const auto &leaf : final_leaves) {
    if (leaf->type == NodeType::MERGE && !leaf->dirty) {
      auto merge_node = std::dynamic_pointer_cast<MergeNode>(leaf);
      if (merge_node) {
        py::object result = merge_node->getResult();
        if (!result.is_none()) {
          predictions.push_back(result);
          found_merge_result = true;
        }
      }
    }
  }

  // Fallback: collect from PREDICT nodes that have results
  if (!found_merge_result) {
    for (const auto &node : m_nodes) {
      if (node->type == NodeType::PREDICT && node->getOutputEdge() &&
          node->getOutputEdge()->isReady()) {
        auto input_data = node->getOutputEdge()->getData();
        if (input_data) {
          // Extract the predictions (y) from the InputNode
          py::object prediction_result = input_data->getY();
          if (!prediction_result.is_none()) {
            predictions.push_back(prediction_result);
          }
        }
      }
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
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  m_nodes.push_back(input_as_node);
  m_node_map[m_input_node->name] = input_as_node;
  m_first_node = input_as_node;
}

void Graph::enableParallelExecution(bool enable) {
  m_parallel_enabled = enable;
}

void Graph::setMulticoreThreshold(size_t threshold) {
  m_multicore_threshold = threshold;
}

void Graph::runParallel() {
  std::cout << "DEBUG: runParallel() called" << std::endl;

  if (!m_parallel_enabled) {
    std::cout << "DEBUG: Parallel not enabled, using sequential" << std::endl;
    run();
    return;
  }

  if (!m_compiled) {
    std::cout << "DEBUG: Graph not compiled, compiling..." << std::endl;
    compile();
  }

  // Check if we have enough nodes AND they're compute-heavy enough to justify
  // parallel overhead
  std::cout << "DEBUG: Execution order size: " << m_execution_order.size()
            << ", threshold: " << m_multicore_threshold << std::endl;
  if (m_execution_order.size() < m_multicore_threshold) {
    std::cout
        << "DEBUG: Not enough nodes for parallel execution, using sequential"
        << std::endl;
    run();
    return;
  }

  // For small/fast operations, threading overhead > benefit, so use sequential
  // Only use parallel for genuinely compute-intensive operations
  bool has_heavy_computation = false;
  for (const auto &node : m_execution_order) {
    if ((node->type == NodeType::MODEL || node->type == NodeType::PREPROCESS) &&
        node->dirty) {
      has_heavy_computation = true;
      break;
    }
  }

  std::cout << "DEBUG: Has heavy computation: " << has_heavy_computation
            << std::endl;
  if (!has_heavy_computation) {
    std::cout << "DEBUG: No heavy computation detected, using sequential"
              << std::endl;
    run();
    return;
  }

  std::cout << "DEBUG: Starting parallel execution..." << std::endl;

  // Group nodes by execution levels (respecting dependencies)
  auto execution_levels = groupNodesByLevel();

  // Execute each level in sequence, but nodes within each level in parallel
  for (const auto &level : execution_levels) {
    std::cout << "DEBUG: Executing level with " << level.size() << " nodes"
              << std::endl;
    if (level.size() == 1) {
      // Single node - execute sequentially
      auto &node = level[0];
      std::cout << "DEBUG: Single node execution: " << node->name << std::endl;
      if (node->dirty) {
        try {
          node->execute();
          node->dirty = false;
        } catch (const std::exception &e) {
          throw std::runtime_error("Error executing node '" + node->name +
                                   "': " + std::string(e.what()));
        }
      }
    } else {
      // Multiple nodes - execute in parallel
      std::cout << "DEBUG: Parallel execution of " << level.size() << " nodes"
                << std::endl;
      // Release GIL for the main thread so worker threads can acquire it
      py::gil_scoped_release release;
      executeNodesInParallel(level);
      // GIL is automatically re-acquired when release goes out of scope
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
        throw std::runtime_error("Error executing node '" + node->name +
                                 "': " + std::string(e.what()));
      }
    }
  }
}

// Accessors
const std::vector<Node::Ptr> &Graph::getNodes() const { return m_nodes; }

bool Graph::isCompiled() const { return m_compiled; }

const std::vector<py::object> &Graph::getPreprocessingFunctions() const {
  return m_preprocessing_functions;
}

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

  // Visit all dependent nodes (nodes that consume this node's output)
  // In the new architecture, each node has only one output edge
  if (node->getOutputEdge()) {
    // Find all nodes that use this node's output as their input
    for (const auto &consumer : m_nodes) {
      if (consumer->getInputEdge() == node->getOutputEdge()) {
        topologicalSortDFS(consumer, visited, temp_visited, result);
      }
    }
  }

  temp_visited.erase(node.get());
  visited.insert(node.get());
  result.push_back(node);
}

void Graph::optimizeGraph() {
  // Resources
  // https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/docs/internal_cpu_plugin_optimization.md
  // https://github.com/openvinotoolkit/openvino/tree/master/src/common/transformations
  passes::NodeFusion::apply(*this);
}

std::vector<Node::Ptr> Graph::findLeafNodes() const {
  std::vector<Node::Ptr> leaves;

  for (const auto &node : m_nodes) {
    bool isLeaf = true;

    // Check if this node's output edge is connected to any other node
    if (node->getOutputEdge()) {
      // Check if any other node uses this node's output edge as input
      for (const auto &other_node : m_nodes) {
        if (other_node != node &&
            other_node->getInputEdge() == node->getOutputEdge()) {
          isLeaf = false;
          break;
        }
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

  if (m_branch_tails.size() < 2) {
    throw std::runtime_error("Need at least 2 branches to merge");
  }

  // Get all branch tail nodes
  std::vector<Node::Ptr> tail_nodes;
  for (const auto &tail_name : m_branch_tails) {
    Node::Ptr tail_node = findNodeByName(tail_name);
    if (tail_node) {
      tail_nodes.push_back(tail_node);
    }
  }

  if (tail_nodes.size() != m_branch_tails.size()) {
    throw std::runtime_error("Some branch tail nodes not found");
  }

  // Binary tree merging: pair nodes and merge them level by level
  std::vector<Node::Ptr> current_level = tail_nodes;
  int level = 0;

  while (current_level.size() > 1) {
    std::vector<Node::Ptr> next_level;

    // Pair up nodes in the current level
    for (size_t i = 0; i < current_level.size(); i += 2) {
      if (i + 1 < current_level.size()) {
        // We have a pair - create merge node
        std::string merge_step_name;
        if (current_level.size() == 2 && level == 0) {
          // If we only have 2 nodes at the first level, use the final name
          merge_step_name = merge_name;
        } else {
          merge_step_name = merge_name + "_step_" + std::to_string(level) +
                            "_" + std::to_string(i / 2);
        }

        Node::Ptr merge_node = NodeFactory::createNode(
            NodeType::MERGE, merge_step_name, merge_func);

        // Add directly to graph without going through addNode
        m_nodes.push_back(merge_node);
        m_node_map[merge_node->name] = merge_node;

        // Connect the pair to this merge node
        current_level[i]->connectTo(merge_node);
        current_level[i + 1]->connectTo(merge_node);

        next_level.push_back(merge_node);
      } else {
        // Odd node - carry it forward to next level
        next_level.push_back(current_level[i]);
      }
    }

    current_level = next_level;
    level++;
  }

  // Rename the final merge node to the requested name if it's not already
  if (current_level.size() == 1 && current_level[0]->name != merge_name) {
    // Update the node map
    m_node_map.erase(current_level[0]->name);
    current_level[0]->name = merge_name;
    m_node_map[merge_name] = current_level[0];
  }

  m_is_branched = false;
  m_branch_tails.clear();
  m_sequential_nodes.push_back(merge_name);
  m_compiled = false; // Mark for recompilation
}

std::vector<std::vector<Node::Ptr>> Graph::extractBranches() const {
  std::vector<std::vector<Node::Ptr>> branches;
  std::unordered_set<Node *> visited;

  // Find all nodes with multiple outgoing connections (branch points)
  std::vector<Node::Ptr> branch_points;
  for (const auto &node : m_nodes) {
    int output_count = 0;
    if (node->getOutputEdge()) {
      // Count how many nodes use this node's output
      for (const auto &other_node : m_nodes) {
        if (other_node != node &&
            other_node->getInputEdge() == node->getOutputEdge()) {
          output_count++;
        }
      }
    }
    if (output_count > 1) {
      branch_points.push_back(node);
    }
  }

  // For each branch point, trace all paths
  for (const auto &branch_point : branch_points) {
    if (branch_point->getOutputEdge()) {
      for (const auto &consumer : m_nodes) {
        if (consumer->getInputEdge() == branch_point->getOutputEdge() &&
            visited.find(consumer.get()) == visited.end()) {

          std::vector<Node::Ptr> branch;
          Node::Ptr current = consumer;

          // Trace the branch until we hit a merge point or leaf
          while (current && visited.find(current.get()) == visited.end()) {
            branch.push_back(current);
            visited.insert(current.get());

            // Find next node in this branch
            Node::Ptr next = nullptr;
            if (current->getOutputEdge()) {
              for (const auto &next_consumer : m_nodes) {
                if (next_consumer->getInputEdge() == current->getOutputEdge()) {
                  next = next_consumer;
                  break;
                }
              }
            }
            current = next;
          }

          if (!branch.empty()) {
            branches.push_back(branch);
          }
        }
      }
    }
  }

  return branches;
}

std::vector<std::vector<Node::Ptr>> Graph::groupNodesByLevel() const {
  std::vector<std::vector<Node::Ptr>> levels;
  std::unordered_map<Node *, int> node_levels;
  std::unordered_set<Node *> processed;

  // Calculate the level of each node based on dependencies
  std::function<int(Node::Ptr)> calculateLevel = [&](Node::Ptr node) -> int {
    if (node_levels.find(node.get()) != node_levels.end()) {
      return node_levels[node.get()];
    }

    int max_dependency_level = -1;

    // Find all nodes that this node depends on
    if (node->getInputEdge()) {
      for (const auto &dependency : m_nodes) {
        if (dependency->getOutputEdge() == node->getInputEdge()) {
          max_dependency_level =
              std::max(max_dependency_level, calculateLevel(dependency));
          break;
        }
      }
    }

    int level = max_dependency_level + 1;
    node_levels[node.get()] = level;
    return level;
  };

  // Calculate levels for all nodes
  int max_level = 0;
  for (const auto &node : m_execution_order) {
    int level = calculateLevel(node);
    max_level = std::max(max_level, level);
  }

  // Group nodes by level
  levels.resize(max_level + 1);
  for (const auto &node : m_execution_order) {
    int level = node_levels[node.get()];
    levels[level].push_back(node);
  }

  // Remove empty levels
  levels.erase(std::remove_if(levels.begin(), levels.end(),
                              [](const std::vector<Node::Ptr> &level) {
                                return level.empty();
                              }),
               levels.end());

  return levels;
}

void Graph::executeNodesInParallel(const std::vector<Node::Ptr> &nodes) {
  std::vector<std::future<void>> futures;
  std::vector<std::exception_ptr> exceptions(nodes.size());

  // Launch parallel execution for each node
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (node->dirty) {
      futures.emplace_back(
          std::async(std::launch::async, [node, &exceptions, i]() {
            try {
              // Acquire GIL for this thread since we're working with Python
              // objects
              py::gil_scoped_acquire acquire;
              node->execute();
              node->dirty = false;
            } catch (...) {
              exceptions[i] = std::current_exception();
            }
          }));
    }
  }

  // Wait for all futures to complete
  for (auto &future : futures) {
    future.wait();
  }

  // Check for exceptions and rethrow the first one found
  for (size_t i = 0; i < exceptions.size(); ++i) {
    if (exceptions[i]) {
      try {
        std::rethrow_exception(exceptions[i]);
      } catch (const std::exception &e) {
        throw std::runtime_error("Error executing node '" + nodes[i]->name +
                                 "' in parallel: " + std::string(e.what()));
      }
    }
  }
}

} // namespace mainera