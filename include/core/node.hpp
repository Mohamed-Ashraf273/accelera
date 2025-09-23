#ifndef NODE_HPP
#define NODE_HPP

#include "core/visibility.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

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
};

class MAINERA_API Node : public std::enable_shared_from_this<Node> {
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
  std::shared_ptr<Node> getSourceNode() const { return m_sourceNode; }
  void setSourceNode(std::shared_ptr<Node> source) { m_sourceNode = source; }

  // Memory optimization methods
  void setShouldCreateNewData(bool should_create);
  bool getShouldCreateNewData() const;

  std::shared_ptr<InputNode> getInput();
  void setOutput(std::shared_ptr<InputNode> result);
  std::shared_ptr<InputNode> getOutput();

  // GPU handling methods
  void usesGPU();
  void setUsesGPU(bool uses_gpu) { m_uses_gpu = uses_gpu; }
  bool getUsesGPU() const { return m_uses_gpu; }

protected:
  std::shared_ptr<InputNode> m_data;
  Graph *m_graph = nullptr; // Pointer to parent graph
  std::shared_ptr<Node> m_sourceNode = nullptr;
  bool m_uses_gpu = false;
};

} // namespace mainera

#endif // NODE_HPP