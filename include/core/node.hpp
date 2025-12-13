#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "core/visibility.hpp"

namespace py = pybind11;

namespace accelera {

class Graph;
class InputNode;

enum class NodeType {
  INPUT,
  PREPROCESS,
  FEATURE,
  MODEL,
  PREDICT,
  MERGE,
  METRIC
};

class ACCELERA_API Node : public std::enable_shared_from_this<Node> {
public:
  using Ptr = std::shared_ptr<Node>;
  NodeType type;
  std::string name;
  py::object py_func;
  bool should_create_new_data = false;
  bool selected_in_path = false;

  Node(NodeType type, const std::string &name, py::object py_func);
  virtual ~Node() = default;

  Ptr clone() const;

  void setGraph(Graph *graph);
  Graph *getGraph() const;

  virtual void execute() = 0;

  void setData(std::shared_ptr<py::object> input);
  std::shared_ptr<py::object> getData() const;

  std::shared_ptr<Node> getSourceNode() const;
  const std::vector<std::shared_ptr<Node>> getSourceNodes() const;
  void setSourceNode(std::shared_ptr<Node> source);
  void setSourceNodes(const std::vector<std::shared_ptr<Node>> &source_nodes);

  void setShouldCreateNewData(bool should_create);
  bool getShouldCreateNewData() const;

  void usesGPU();

  void setUsesGPU(bool uses_gpu);
  bool getUsesGPU() const;

protected:
  std::shared_ptr<py::object> m_data = std::make_shared<py::object>(py::none());
  Graph *m_graph = nullptr; // Pointer to parent graph
  std::vector<std::shared_ptr<Node>> m_sourceNode;
  bool m_uses_gpu = false;
};

} // namespace accelera

#endif // NODE_HPP