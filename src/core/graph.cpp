#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/merge.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace aistudio {

Graph::Graph()
    : m_compiled(false), m_parallel_enabled(false), m_first_node(nullptr) {
  // Automatically create an input node as the starting point
  m_input_node = std::make_shared<InputNode>();
  m_nodes.push_back(m_input_node);
  m_node_map[m_input_node->name] = m_input_node;
  m_first_node = m_input_node;
}

Graph::~Graph() { clear(); }

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

        // Determine appropriate input/output counts based on node type
        size_t numInputs, numOutputs;
        if (node->type == NodeType::PREPROCESS) {
          numInputs = 2;  // X and y
          numOutputs = 2; // processed X and y
        } else if (node->type == NodeType::MODEL) {
          numInputs = 2;  // X and y
          numOutputs = 1; // trained model
        } else {
          // For other node types, use the original node's edge counts
          numInputs = node->getInputEdges().size();
          numOutputs = node->getOutputEdges().size();
        }

        nodeToAdd = NodeFactory::createNode(node->type, copyName, numInputs,
                                            numOutputs, node->py_func);
      }

      // Add the node to the graph
      m_nodes.push_back(nodeToAdd);
      m_node_map[nodeToAdd->name] = nodeToAdd; // Fast lookup

      // Connect the leaf to this node
      // Handle multiple input/output connections for nodes that need both X and
      // y
      if (!leaves[i]->getOutputEdges().empty() &&
          !nodeToAdd->getInputEdges().empty()) {

        // Connect first output to first input (X connection)
        leaves[i]->connectTo(0, nodeToAdd, 0);

        // If both leaf and node have second edges, connect them (y connection)
        if (leaves[i]->getOutputEdges().size() > 1 &&
            nodeToAdd->getInputEdges().size() > 1) {
          leaves[i]->connectTo(1, nodeToAdd, 1);
        }
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

      // Determine appropriate input/output counts based on node type
      size_t numInputs, numOutputs;
      if (nodeType == NodeType::PREPROCESS) {
        numInputs = 2;  // X and y
        numOutputs = 2; // processed X and y
      } else if (nodeType == NodeType::MODEL) {
        numInputs = 2;  // X and y
        numOutputs = 1; // trained model
      } else {
        numInputs = 1;  // Default
        numOutputs = 1; // Default
      }

      // Create the new node
      Node::Ptr branchNode =
          NodeFactory::createNode(nodeType, uniqueName, numInputs, numOutputs,
                                  branch_objects[node_idx]);

      // Add to graph
      m_nodes.push_back(branchNode);
      m_node_map[branchNode->name] = branchNode;

      // Connect the leaf to this branch node
      if (!leaf->getOutputEdges().empty() &&
          !branchNode->getInputEdges().empty()) {
        // Connect both outputs from leaf to both inputs of branch node
        if (leaf->getOutputEdges().size() >= 2 &&
            branchNode->getInputEdges().size() >= 2) {
          leaf->connectTo(0, branchNode, 0); // X connection
          leaf->connectTo(1, branchNode, 1); // y connection
        } else {
          leaf->connectTo(0, branchNode, 0); // Single connection
        }
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
      // Cast to MergeNode and get the result
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
      if (node->type == NodeType::PREDICT && !node->getOutputEdges().empty() &&
          node->getOutputEdges()[0]->isReady()) {
        predictions.push_back(node->getOutputEdges()[0]->getData());
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
  m_nodes.push_back(m_input_node);
  m_node_map[m_input_node->name] = m_input_node;
  m_first_node = m_input_node;
}

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
  // run parallel
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
            NodeType::MERGE, merge_step_name, 2, 1, merge_func);

        // Add directly to graph without going through addNode
        m_nodes.push_back(merge_node);
        m_node_map[merge_node->name] = merge_node;

        // Connect the pair to this merge node
        current_level[i]->connectTo(0, merge_node, 0);
        current_level[i + 1]->connectTo(0, merge_node, 1);

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

} // namespace aistudio