#ifndef NODE_HPP
#define NODE_HPP

#include "edge.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace mainera {

class Graph;     // Forward declaration
class InputNode; // Forward declaration

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

  Node(NodeType type, const std::string &name, py::object py_func);
  virtual ~Node();

  // Graph access
  void setGraph(Graph *graph);
  Graph *getGraph() const;

  // Public members for Python access
  NodeType type;
  std::string name;
  py::object py_func;
  bool dirty = true;
  bool should_create_new_data = false;

  virtual void execute() = 0;

  // Graph connectivity methods
  const std::shared_ptr<Edge> &getInputEdge() const;
  const std::shared_ptr<Edge> &getOutputEdge() const;
  void connectTo(Node::Ptr targetNode);

  // Memory optimization methods
  void setShouldCreateNewData(bool should_create);
  bool getShouldCreateNewData() const;

protected:
  py::list collectInputs();
  void setOutputs(py::object result);
  std::shared_ptr<InputNode> getInput();
  void setOutput(std::shared_ptr<InputNode> result);

  std::shared_ptr<Edge> m_inputEdge;
  std::shared_ptr<Edge> m_outputEdge;
  Graph *m_graph = nullptr; // Pointer to parent graph
};

} // namespace mainera

#endif // NODE_HPP