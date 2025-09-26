#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "core/visibility.hpp"

namespace py = pybind11;

namespace mainera {

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

class MAINERA_API Node : public std::enable_shared_from_this<Node> {
public:
  using Ptr = std::shared_ptr<Node>;
  NodeType type;
  std::string name;
  py::object py_func;
  bool should_create_new_data = false;

  Node(NodeType type, const std::string &name, py::object py_func);
  virtual ~Node() = default;

  Ptr clone() const;

  void setGraph(Graph *graph);
  Graph *getGraph() const;

  virtual void execute() = 0;

  void setData(py::object input);
  py::object getData() const;

  std::shared_ptr<Node> getSourceNode() const;
  void setSourceNode(std::shared_ptr<Node> source);

  void setShouldCreateNewData(bool should_create);
  bool getShouldCreateNewData() const;

  void usesGPU();

  void setUsesGPU(bool uses_gpu);
  bool getUsesGPU() const;

protected:
  py::object m_data = py::none();
  Graph *m_graph = nullptr;
  std::shared_ptr<Node> m_sourceNode = nullptr;
  bool m_uses_gpu = false;
};

} // namespace mainera

#endif // NODE_HPP