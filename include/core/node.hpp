#ifndef NODE_HPP
#define NODE_HPP

#include <pybind11/pybind11.h>

#include "core/visibility.hpp"

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace mainera {

class Graph;
class InputNode;

enum class NodeType { INPUT, PREPROCESS, FEATURE, MODEL, PREDICT, MERGE, METRIC };

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

    void setData(py::object input);
    py::object getData() const;

    // Graph connectivity methods
    std::shared_ptr<Node> getSourceNode() const;
    void setSourceNode(std::shared_ptr<Node> source);

    // Memory optimization methods
    void setShouldCreateNewData(bool should_create);
    bool getShouldCreateNewData() const;

    // GPU handling methods
    void usesGPU();
    void setUsesGPU(bool uses_gpu);
    bool getUsesGPU() const;

protected:
    py::object m_data = py::none();
    Graph *m_graph = nullptr; // Pointer to parent graph
    std::shared_ptr<Node> m_sourceNode = nullptr;
    bool m_uses_gpu = false;
};

} // namespace mainera

#endif // NODE_HPP