#ifndef NODE_HPP
#define NODE_HPP

#include "edge.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace aistudio {

class Graph; // Forward declaration

enum class NodeType {
  INPUT,
  PREPROCESS,
  FEATURE,
  MODEL,
  PREDICT,
  MERGE,
};

class __attribute__((visibility("default"))) Node
    : public std::enable_shared_from_this<Node> {
public:
  using Ptr = std::shared_ptr<Node>;

  Node(NodeType type, const std::string &name, size_t numInputs,
       size_t numOutputs, py::object py_func);
  virtual ~Node();

  // Graph access
  void setGraph(Graph *graph);
  Graph *getGraph() const;

  // Public members for Python access
  NodeType type;
  std::string name;
  py::object py_func;
  bool dirty = true;

  virtual void execute() = 0;

  // Graph connectivity methods
  const std::vector<std::shared_ptr<Edge>> &getInputEdges() const;
  const std::vector<std::shared_ptr<Edge>> &getOutputEdges() const;
  void connectTo(size_t myOutputIndex, Node::Ptr targetNode,
                 size_t targetInputIndex);

protected:
  py::list collectInputs();
  void setOutputs(py::object result);

  std::vector<std::shared_ptr<Edge>> m_inputEdges;
  std::vector<std::shared_ptr<Edge>> m_outputEdges;
  Graph *m_graph = nullptr; // Pointer to parent graph
};

} // namespace aistudio

#endif // NODE_HPP