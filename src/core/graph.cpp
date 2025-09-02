#include "core/graph.hpp"
#include "core/node_factory.hpp"
#include "nodes/branch.hpp"
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace aistudio {

Graph::Graph()
    : m_compiled(false), m_parallel_enabled(false), m_first_node(nullptr) {}

Graph::~Graph() { clear(); }

Node::Ptr Graph::add_node(NodeType type, const std::string &name,
                          py::object py_func, size_t num_inputs,
                          size_t num_outputs) {
  auto node =
      NodeFactory::createNode(type, name, num_inputs, num_outputs, py_func);
  addNode(node);
  return node;
}

Node::Ptr Graph::add_branch_node(const std::string &name,
                                 const std::vector<Node::Ptr> &branches) {
  // Create a branch node that fans out to multiple paths
  auto branch_node =
      std::make_shared<BranchNode>(name, 2, branches.size() * 2, py::none());
  addNode(branch_node);

  // Connect branch node outputs to all branch models (X and y for each)
  for (size_t i = 0; i < branches.size(); ++i) {
    branch_node->connectTo(i * 2, branches[i], 0);     // X to model
    branch_node->connectTo(i * 2 + 1, branches[i], 1); // y to model
  }

  return branch_node;
}

Node::Ptr Graph::createBranch(const std::string &name,
                              const std::vector<Node::Ptr> &models,
                              py::object merge_func) {
  // Create a branch node with proper input/output configuration
  size_t num_outputs = models.size() * 2; // Each model needs X and y
  auto branch_node =
      std::make_shared<BranchNode>(name, 2, num_outputs, merge_func);
  addNode(branch_node);

  // Connect branch outputs to models
  for (size_t i = 0; i < models.size(); ++i) {
    branch_node->connectTo(i * 2, models[i], 0);     // X to model input 0
    branch_node->connectTo(i * 2 + 1, models[i], 1); // y to model input 1
  }

  return branch_node;
}

std::vector<Node::Ptr>
Graph::duplicateNodeForBranches(Node::Ptr template_node,
                                const std::vector<Node::Ptr> &branch_models) {
  std::vector<Node::Ptr> duplicated_nodes;

  for (size_t i = 0; i < branch_models.size(); ++i) {
    // Create a new node for each branch
    std::string new_name =
        template_node->name + "_branch" + std::to_string(i + 1);

    // Determine number of inputs/outputs based on node type
    size_t num_inputs = template_node->getInputEdges().size();
    size_t num_outputs = template_node->getOutputEdges().size();

    auto new_node =
        NodeFactory::createNode(template_node->type, new_name, num_inputs,
                                num_outputs, template_node->py_func);
    addNode(new_node);

    // Connect the branch model to this new node
    if (template_node->type == NodeType::PREDICT) {
      // For predict nodes: model output → predict input 1, test data → predict
      // input 0
      branch_models[i]->connectTo(0, new_node, 1);
    } else {
      // For other nodes: normal connection
      branch_models[i]->connectTo(0, new_node, 0);
    }

    duplicated_nodes.push_back(new_node);
  }

  return duplicated_nodes;
}

std::vector<py::object>
Graph::collectBranchResults(const std::vector<Node::Ptr> &result_nodes) {
  std::vector<py::object> results;

  for (size_t i = 0; i < result_nodes.size(); ++i) {
    if (result_nodes[i]->getOutputEdges().size() > 0 &&
        result_nodes[i]->getOutputEdges()[0]->isReady()) {
      py::object result = result_nodes[i]->getOutputEdges()[0]->getData();
      results.push_back(result);
    } else {
      results.push_back(py::none());
    }
  }

  return results;
}

std::vector<Node::Ptr> Graph::createBranchPipeline(
    const std::string &branch_name, const std::vector<Node::Ptr> &models,
    const std::string &predict_name, py::object test_data) {
  // Create branch node with 2 inputs (X, y) and outputs for each model
  size_t num_outputs = models.size() * 2; // Each model needs X and y
  auto branch_node =
      std::make_shared<BranchNode>(branch_name, 2, num_outputs, py::none());
  addNode(branch_node);

  // Connect branch outputs to models
  for (size_t i = 0; i < models.size(); ++i) {
    branch_node->connectTo(i * 2, models[i], 0);     // X to model input 0
    branch_node->connectTo(i * 2 + 1, models[i], 1); // y to model input 1
  }

  // Create predict nodes for each model
  std::vector<Node::Ptr> predict_nodes;
  for (size_t i = 0; i < models.size(); ++i) {
    std::string pred_name = predict_name + "_branch" + std::to_string(i + 1);
    auto predict_node =
        NodeFactory::createNode(NodeType::PREDICT, pred_name, 2, 1, py::none());
    addNode(predict_node);

    // Connect model output to predict input 1
    models[i]->connectTo(0, predict_node, 1);

    // Set test data on predict input 0 if provided
    if (!test_data.is_none()) {
      predict_node->getInputEdges()[0]->setData(test_data);
    }

    predict_nodes.push_back(predict_node);
  }

  // Set branch as first node
  setFirstNode(branch_node);

  return predict_nodes;
}

void Graph::setFirstNode(Node::Ptr node) { m_first_node = node; }

Node::Ptr Graph::getFirstNode() const {
  // Simple implementation - return the first node if available
  return m_nodes.empty() ? nullptr : m_nodes[0];
}

void Graph::enableParallelExecution(bool enable) {
  // Set flag for parallel execution
  m_parallel_enabled = enable;
}

void Graph::runParallel() {
  if (!m_parallel_enabled) {
    run(); // Fall back to sequential execution
    return;
  }

  // Parallel execution implementation
  auto sorted_nodes = topologicalSort();

  // Group nodes by execution level (nodes that can run in parallel)
  std::vector<std::vector<Node::Ptr>> execution_levels;
  std::vector<int> node_levels(sorted_nodes.size(), -1);

  // Calculate execution levels
  for (size_t i = 0; i < sorted_nodes.size(); ++i) {
    int max_input_level = -1;
    auto &node = sorted_nodes[i];

    // Find the maximum level of input dependencies
    for (auto &input_edge : node->getInputEdges()) {
      if (input_edge) {
        // Find which node provides this input
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

    // Add to execution level
    if (current_level >= (int)execution_levels.size()) {
      execution_levels.resize(current_level + 1);
    }
    execution_levels[current_level].push_back(node);
  }

  // Execute nodes level by level with parallelism within each level
  for (auto &level : execution_levels) {
    if (level.size() == 1) {
      // Single node - execute normally with GIL
      try {
        py::gil_scoped_acquire acquire; // Ensure GIL for Python object access
        level[0]->execute();
      } catch (const std::exception &e) {
        std::cerr << "Error executing node '" << level[0]->name
                  << "': " << e.what() << std::endl;
        throw;
      }
    } else {
      // Multiple nodes - execute in parallel using threads with GIL handling
      std::vector<std::thread> threads;
      std::vector<std::exception_ptr> exceptions(level.size());

      for (size_t i = 0; i < level.size(); ++i) {
        threads.emplace_back([&level, &exceptions, i]() {
          try {
            // Acquire GIL for this thread when working with Python objects
            py::gil_scoped_acquire acquire;
            level[i]->execute();
          } catch (...) {
            exceptions[i] = std::current_exception();
          }
        });
      }

      // Release GIL while waiting for threads to complete
      {
        py::gil_scoped_release release;

        // Wait for all threads to complete
        for (auto &thread : threads) {
          thread.join();
        }
      }

      // Check for exceptions
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

void Graph::addNode(Node::Ptr node) {
  m_nodes.push_back(node);
  m_compiled = false; // Graph needs recompilation

  // Track first node for input provision - prefer BRANCH nodes over MODEL nodes
  if (!m_first_node) {
    if (node->type == NodeType::BRANCH) {
      m_first_node = node;
    } else if (node->type == NodeType::MODEL) {
      m_first_node = node;
    }
  } else if (node->type == NodeType::BRANCH &&
             m_first_node->type == NodeType::MODEL) {
    // Replace MODEL with BRANCH node as first node (BRANCH has higher priority)
    m_first_node = node;
  }
}

void Graph::compile() {
  if (m_compiled)
    return;

  m_execution_order = topologicalSort();

  optimizeGraph();

  m_compiled = true;
}

void Graph::run() {
  if (!m_compiled) {
    compile();
  }

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

std::vector<py::object> Graph::execute(py::object X, py::object y) {
  if (!m_compiled) {
    compile();
  }

  // Provide inputs to first node
  if (m_first_node) {
    if (m_first_node->getInputEdges().size() >= 2) {
      m_first_node->getInputEdges()[0]->setData(X);
      m_first_node->getInputEdges()[1]->setData(y);
    } else if (m_first_node->getInputEdges().size() == 1) {
      m_first_node->getInputEdges()[0]->setData(X);
    }
  }

  // Execute nodes - use parallel execution if enabled
  if (m_parallel_enabled) {
    runParallel();
  } else {
    // Execute all nodes in order
    for (auto &node : m_execution_order) {
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

  // Collect outputs from prediction nodes
  std::vector<py::object> predictions;
  for (auto &node : m_nodes) {
    if (node->type == NodeType::PREDICT) {
      if (node->getOutputEdges().size() > 0 &&
          node->getOutputEdges()[0]->isReady()) {
        predictions.push_back(node->getOutputEdges()[0]->getData());
      }
    }
  }

  return predictions;
}

void Graph::autoConnectNodes() {
  // Simple linear auto-connection
  for (size_t i = 0; i < m_nodes.size() - 1; ++i) {
    auto &current = m_nodes[i];
    auto &next = m_nodes[i + 1];

    if (current->getOutputEdges().size() > 0 &&
        next->getInputEdges().size() > 0) {
      current->connectTo(0, next, 0);
    }
  }
}

void Graph::clear() {
  m_nodes.clear();
  m_execution_order.clear();
  m_compiled = false;
}

void Graph::set_input(Node::Ptr node, size_t input_index, py::object data) {
  if (input_index >= node->getInputEdges().size()) {
    throw std::out_of_range("Input index out of range for node: " + node->name);
  }
  node->getInputEdges()[input_index]->setData(data);
}

py::object Graph::get_output(Node::Ptr node, size_t output_index) {
  if (output_index >= node->getOutputEdges().size()) {
    throw std::out_of_range("Output index out of range for node: " +
                            node->name);
  }
  return node->getOutputEdges()[output_index]->getData();
}

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
  std::cout << "Graph optimization placeholder" << std::endl;
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

std::vector<Node::Ptr>
Graph::createPredictForBranches(const std::string &predict_name,
                                const std::vector<Node::Ptr> &models,
                                py::object test_data) {
  std::vector<Node::Ptr> predict_nodes;

  for (size_t i = 0; i < models.size(); ++i) {
    std::string pred_name = predict_name + "_branch" + std::to_string(i + 1);
    auto predict_node =
        NodeFactory::createNode(NodeType::PREDICT, pred_name, 2, 1, py::none());
    addNode(predict_node);

    // Connect model output to predict input 1
    models[i]->connectTo(0, predict_node, 1);

    // Set test data on predict input 0 if provided
    if (!test_data.is_none()) {
      predict_node->getInputEdges()[0]->setData(test_data);
    }

    predict_nodes.push_back(predict_node);
  }

  return predict_nodes;
}

} // namespace aistudio