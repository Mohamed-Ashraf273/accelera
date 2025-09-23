#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/input.hpp"
#include "nodes/merge.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"
#include "nodes/preprocess.hpp"
#include "utils/graph_utils.hpp"
#include <iostream>

namespace mainera {

Graph::Graph() : m_compiled(false), m_parallel_enabled(false) {
  // Automatically create an input node as the starting point
  m_input_node = std::make_shared<InputNode>();
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  input_as_node->setSourceNode(nullptr);
  m_nodes.push_back(input_as_node);
}

Graph::~Graph() { clear(); }

Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func) {
  auto node = NodeFactory::createNode(type, name, py_func);

  addNode(node);
  return node;
}

void Graph::addNode(Node::Ptr node) {
  node->setGraph(this);

  std::vector<Node::Ptr> leaves = findLeafNodes();

  // Create a copy of the node for each leaf and connect them
  std::vector<std::string>
      new_branch_tails; // Track new tails if in branched mode

  for (size_t i = 0; i < leaves.size(); ++i) {
    Node::Ptr nodeToAdd;

    if (i == 0) {
      nodeToAdd = node;
    } else {
      // Create a copy for subsequent leaves
      std::string copyName = node->name + "_copy_" + std::to_string(i);
      nodeToAdd = NodeFactory::createNode(node->type, copyName, node->py_func);
      // Set graph pointer for copied node
      nodeToAdd->setGraph(this);
    }

    m_nodes.push_back(nodeToAdd);

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
    nodeToAdd->setSourceNode(leaves[i]);

    // If in branched mode, track the new node as a potential branch tail
    if (m_is_branched) {
      new_branch_tails.push_back(nodeToAdd->name);
    }
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

  // Clear any previous branch state
  m_is_branched = true;

  // For each leaf, create parallel branches
  for (size_t leaf_idx = 0; leaf_idx < leaves.size(); ++leaf_idx) {
    Node::Ptr leaf = leaves[leaf_idx];

    // Store the tail nodes for each parallel branch
    std::vector<Node::Ptr> branch_tails;

    for (size_t branch_idx = 0; branch_idx < branch_objects.size();
         ++branch_idx) {
      Node::Ptr current_source = leaf;

      // Check if this branch object is a list of nodes
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

          // Only the first node in each branch should create new data
          branchNode->setShouldCreateNewData(list_idx == 0);

          m_nodes.push_back(branchNode);

          // Connect to the current source in the chain
          branchNode->setSourceNode(current_source);
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
        branchNode->setShouldCreateNewData(true);

        m_nodes.push_back(branchNode);

        branchNode->setSourceNode(leaf);
        current_source = branchNode;
      }

      // Store the final node of this branch
      branch_tails.push_back(current_source);
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

std::vector<py::object> Graph::execute(py::object X, py::object y) {
  if (!m_compiled)
    compile();

  if (m_input_node) {
    m_input_node->setInputData(X, y);
    m_input_node->setOutput(m_input_node);
  }

  if (m_parallel_enabled && m_execution_order.size() >= m_multicore_threshold) {
    runParallel();
  } else {
    run();
  }

  // Collect results from leaf nodes
  std::vector<py::object> predictions;
  std::vector<Node::Ptr> final_leaves = findLeafNodes();

  for (const auto &leaf : final_leaves) {
    if (!leaf->dirty) {
      try {
        if (leaf->type == NodeType::PREDICT) {
          py::object result = leaf->getOutput()->getY();
          if (!result.is_none()) {
            predictions.push_back(result);
          }
        } else if (leaf->type == NodeType::PREPROCESS) {
          py::object result = leaf->getOutput()->getX();
          if (!result.is_none()) {
            predictions.push_back(result);
          }
        } else if (leaf->type == NodeType::MODEL) {
          py::object result = leaf->getOutput()->getFittedModel();
          if (!result.is_none()) {
            predictions.push_back(result);
          }
        } else if (leaf->type == NodeType::INPUT) {
          py::object result = py::make_tuple(leaf->getOutput()->getX(),
                                             leaf->getOutput()->getY());
          if (!result.is_none()) {
            predictions.push_back(result);
          }
        }
        // Fallback for other node types
        else {
          throw std::runtime_error(
              "Unhandled node type for result collection: " +
              std::to_string(static_cast<int>(leaf->type)));
        }
      } catch (const std::exception &e) {
        log_warning("Error collecting results from node '" + leaf->name +
                    "': " + e.what());
      }
    }
  }

  return predictions;
}

void Graph::clear() {
  m_nodes.clear();
  m_execution_order.clear();
  m_compiled = false;
  m_is_branched = false;
  m_input_node = nullptr;

  // Recreate the input node
  m_input_node = std::make_shared<InputNode>();
  auto input_as_node = std::static_pointer_cast<Node>(m_input_node);
  m_nodes.push_back(input_as_node);
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
  // Group nodes by execution levels (respecting dependencies)
  auto execution_levels = groupNodesByLevel();

  // Execute each level in sequence, but nodes within each level in parallel
  for (const auto &level : execution_levels) {
    if (level.size() == 1) {
      // Single node - execute sequentially
      auto &node = level[0];
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

  // Build adjacency list: node -> list of consumers (nodes that have this node
  // as source)
  std::unordered_map<Node *, std::vector<Node::Ptr>> adjacency_list;
  std::unordered_map<Node *, int> in_degrees;

  // Initialize
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
    throw std::runtime_error("Graph contains cycles - not a DAG");
  }

  return sorted_nodes;
}

std::vector<Node::Ptr> Graph::findLeafNodes() const {
  std::vector<Node::Ptr> leaves;

  // Create a set of nodes that serve as sources for other nodes
  std::unordered_set<Node::Ptr> source_nodes;

  for (const auto &node : m_nodes) {
    if (auto source = node->getSourceNode()) {
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

void Graph::mergeBranches(const std::string &merge_name,
                          py::object merge_func) {
  if (!m_is_branched) {
    throw std::runtime_error("Not in branched mode - nothing to merge");
  }

  throw std::runtime_error("Merge functionality not yet implemented");
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
    if (node->dirty) {
      if (node->getUsesGPU()) {
        gpu_nodes.push_back(node);
      } else {
        cpu_nodes.push_back(node);
      }
    }
  }

  std::vector<std::exception_ptr> exceptions(nodes.size());
  std::vector<std::future<void>> futures;

  if (!cpu_nodes.empty()) {
    for (size_t i = 0; i < cpu_nodes.size(); ++i) {
      const auto &node = cpu_nodes[i];
      futures.emplace_back(
          std::async(std::launch::async, [node, &exceptions, i]() {
            py::gil_scoped_acquire acquire; // Acquire GIL at thread start
            try {
              node->execute();
              node->dirty = false;
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
                     py::gil_scoped_acquire
                         acquire; // Acquire GIL for the entire GPU sequence
                     try {
                       for (size_t i = 0; i < gpu_nodes.size(); ++i) {
                         gpu_nodes[i]->execute();
                         gpu_nodes[i]->dirty = false;
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

  // Wait for ALL futures (both CPU and GPU) to complete
  for (auto &future : futures) {
    future.wait();
  }

  // Check for exceptions
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
        failed_node = nodes[i]; // Fallback
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

} // namespace mainera