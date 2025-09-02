#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "node.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aistudio {

class __attribute__((visibility("default"))) Graph {
public:
  Graph();
  ~Graph();

  // Core node management
  Node::Ptr add_node(NodeType type, const std::string &name, py::object py_func,
                     size_t num_inputs = 1, size_t num_outputs = 1);
  void addNode(Node::Ptr node);

  // Graph operations
  void compile();
  std::vector<py::object> execute(py::object X, py::object y);
  void clear();

  // Pipeline management - handle sequential and branched modes
  void addSequentialNode(const std::string &node_type, const std::string &name,
                         py::object obj);
  void startBranching(const std::string &branch_name,
                      const std::vector<py::object> &branch_models);
  void addToBranches(const std::string &node_type, const std::string &name,
                     py::object obj);
  void mergeBranches(const std::string &merge_name, py::object merge_func);
  bool isBranched() const;

  // Parallel execution
  void enableParallelExecution(bool enable = true);

  // Accessors
  const std::vector<Node::Ptr> &getNodes() const;
  bool isCompiled() const;

  // Serialization
  void serialize(const std::string &filepath) const;

private:
  // Core execution methods
  void run();
  void runParallel();

  // Topological sorting
  std::vector<Node::Ptr> topologicalSort();
  void topologicalSortDFS(Node::Ptr node, std::unordered_set<Node *> &visited,
                          std::unordered_set<Node *> &temp_visited,
                          std::vector<Node::Ptr> &result);

  // Utility methods
  void optimizeGraph();
  Node::Ptr findNodeByName(const std::string &name) const;
  bool shouldReplaceFirstNode(Node::Ptr node) const;

  // Core data members
  std::vector<Node::Ptr> m_nodes;
  std::vector<Node::Ptr> m_execution_order;
  std::unordered_map<std::string, Node::Ptr> m_node_map; // Fast node lookup
  bool m_compiled = false;
  bool m_parallel_enabled = false;
  Node::Ptr m_first_node = nullptr;

  // Pipeline state management
  std::vector<std::string> m_sequential_nodes;
  bool m_is_branched = false;
  std::vector<std::string> m_branch_tails;
  int m_node_counter = 0;
};

} // namespace aistudio

#endif // GRAPH_HPP