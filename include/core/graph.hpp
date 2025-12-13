#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/node.hpp"
#include "core/visibility.hpp"
#include "nodes/metric.hpp"
namespace accelera {

class InputNode; // Forward declaration

class ACCELERA_API Graph {
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
                                  py::object best_path,
                                  py::object custom_strategy);
  void clear();

  void split(const std::string &branch_name,
             const std::vector<py::object> &branch_objects,
             const std::vector<std::string> &node_types,
             const std::vector<std::string> &node_names);

  void enableParallelExecution(bool enable = true);
  void setMulticoreThreshold(size_t threshold);

  const std::vector<Node::Ptr> &getNodes() const;
  bool isCompiled() const;

  void setGPUUsage();

  void setIsExecuted(bool executed) { m_executed = executed; }
  bool getIsExecuted() const { return m_executed; }

  std::shared_ptr<InputNode> getInputNode() const { return m_input_node; }

  void enableDisableMetrics(py::object y_true, py::object enable);
  bool savePreprocessedData(const std::string &directory);

  bool save();
  bool load(const std::string &directory);

private:
  std::vector<Node::Ptr> m_execution_order;
  std::vector<Node::Ptr> m_nodes;
  std::vector<std::shared_ptr<MetricNode>> m_metric_nodes;
  bool m_compiled = false;
  bool m_executed = false;
  bool m_is_branched = false;
  bool m_parallel_enabled = false;
  size_t m_multicore_threshold = 3; // Minimum tasks to use multicore
  std::shared_ptr<InputNode> m_input_node = nullptr;

  std::vector<Node::Ptr> findLeafNodes() const;
  std::vector<Node::Ptr> topologicalSort();

  void run();
  void runParallel();
  void resetSelectedPath();
  void setSelectedPath(const std::string &strategy, py::object custom_strategy);
  void selectMaxPath();
  void selectMinPath();
  void selectCustomPath(py::object custom_strategy);
  void selectAllPaths();
  void findMaxMinMetricNode(bool find_max);
  void setPath(std::shared_ptr<MetricNode> best_metric_node);
};
} // namespace accelera

#endif // GRAPH_HPP