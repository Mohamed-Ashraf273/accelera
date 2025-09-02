#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "node.hpp"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace aistudio {

class __attribute__((visibility("default"))) Graph {
public:
  Graph();
  ~Graph();

  Node::Ptr add_node(NodeType type, const std::string &name, py::object py_func,
                     size_t num_inputs = 1, size_t num_outputs = 1);
  Node::Ptr add_branch_node(const std::string &name,
                            const std::vector<Node::Ptr> &branches);
  void addNode(Node::Ptr node);

  Node::Ptr createBranch(const std::string &name,
                         const std::vector<Node::Ptr> &models,
                         py::object merge_func = py::none());
  std::vector<Node::Ptr>
  duplicateNodeForBranches(Node::Ptr template_node,
                           const std::vector<Node::Ptr> &branch_models);
  std::vector<py::object>
  collectBranchResults(const std::vector<Node::Ptr> &result_nodes);

  std::vector<Node::Ptr> createBranchPipeline(
      const std::string &branch_name, const std::vector<Node::Ptr> &models,
      const std::string &predict_name, py::object test_data = py::none());
  void setFirstNode(Node::Ptr node);
  Node::Ptr getFirstNode() const;

  // Parallel execution support
  void enableParallelExecution(bool enable = true);
  void runParallel();

  // Graph operations
  void compile();
  void run();
  std::vector<py::object>
  execute(py::object X,
          py::object y); // Execute with inputs and return predictions
  void autoConnectNodes();
  void clear();

  // Input/output management
  void set_input(Node::Ptr node, size_t input_index, py::object data);
  py::object get_output(Node::Ptr node, size_t output_index = 0);

  // Serialization
  void serialize(const std::string &filepath) const;
  void serializeDot(const std::string &filepath) const;

  // Accessors
  const std::vector<Node::Ptr> &getNodes() const;
  bool isCompiled() const;

  // Branch operations - create predict nodes for existing branches
  std::vector<Node::Ptr>
  createPredictForBranches(const std::string &predict_name,
                           const std::vector<Node::Ptr> &models,
                           py::object test_data = py::none());

  // Pipeline management - handle sequential and branched modes
  void addSequentialNode(const std::string &node_type, const std::string &name,
                         py::object obj);
  void startBranching(const std::string &branch_name,
                      const std::vector<py::object> &branch_models);
  void addToBranches(const std::string &node_type, const std::string &name,
                     py::object obj);
  void mergeBranches(const std::string &merge_name, py::object merge_func);
  bool isBranched() const;

private:
  // Topological sorting
  std::vector<Node::Ptr> topologicalSort();
  void topologicalSortDFS(Node::Ptr node, std::unordered_set<Node *> &visited,
                          std::unordered_set<Node *> &temp_visited,
                          std::vector<Node::Ptr> &result);

  // Graph optimization
  void optimizeGraph();

  std::vector<Node::Ptr> m_nodes;
  std::vector<Node::Ptr> m_execution_order;
  bool m_compiled = false;
  bool m_parallel_enabled = false;  // Parallel execution flag
  Node::Ptr m_first_node = nullptr; // Track first node for input

  // Pipeline state management
  std::vector<std::string> m_sequential_nodes; // Sequential node names
  bool m_is_branched = false;                  // Branched mode flag
  std::vector<std::string> m_branch_tails;     // Current branch tail nodes
  int m_node_counter = 0;                      // For unique naming
};

} // namespace aistudio

#endif // GRAPH_HPP