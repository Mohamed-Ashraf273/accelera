#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "core/visibility.hpp"
#include "node.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mainera {

class InputNode; // Forward declaration

class MAINERA_API Graph {
public:
  Graph();
  ~Graph();

  Graph(const Graph &other);
  Graph *clone() const;

  Node::Ptr add_node(NodeType type, const std::string &name,
                     py::object py_func);
  void addNode(Node::Ptr node);

  void compile();
  std::vector<py::object> execute(py::object X, py::object y,
                                  py::object best_path);
  void clear();

  void split(const std::string &branch_name,
             const std::vector<py::object> &branch_objects,
             const std::vector<std::string> &node_types,
             const std::vector<std::string> &node_names);
  void mergeBranches(const std::string &merge_name, py::object merge_func);

  void enableParallelExecution(bool enable = true);
  void setMulticoreThreshold(size_t threshold);

  const std::vector<Node::Ptr> &getNodes() const;
  bool isCompiled() const;

  std::vector<py::object> getPreprocessingFunctions(Node::Ptr node) const;

  void setGPUUsage();

  void setIsExecuted(bool executed) { m_executed = executed; }
  bool getIsExecuted() const { return m_executed; }

  std::shared_ptr<InputNode> getInputNode() const { return m_input_node; }

  void enableDisableMetrics(py::object y_true, py::object enable);
  bool validateNodeConnection(Node::Ptr newNode, Node::Ptr sourceNode) const;
  std::string nodeTypeToString(NodeType type) const;

private:
  std::vector<Node::Ptr> m_nodes;
  std::vector<Node::Ptr> m_execution_order;
  bool m_compiled = false;
  bool m_parallel_enabled = false;
  bool m_executed = false;
  size_t m_multicore_threshold = 3; // Minimum tasks to use multicore
  std::shared_ptr<InputNode> m_input_node = nullptr; // Automatic input node
  bool m_is_branched = false;

  void run();
  void runParallel();

  std::vector<Node::Ptr> findLeafNodes() const;
  std::vector<Node::Ptr> topologicalSort();
  std::vector<std::vector<Node::Ptr>> groupNodesByLevel() const;
  void executeNodesInParallel(const std::vector<Node::Ptr> &nodes);
};

} // namespace mainera

#endif // GRAPH_HPP