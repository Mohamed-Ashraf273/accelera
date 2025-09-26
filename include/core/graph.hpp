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

    Node::Ptr add_node(NodeType type, const std::string &name, py::object py_func);
    void addNode(Node::Ptr node);

    void compile();
    std::vector<py::object> execute(py::object X, py::object y);
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

private:
    // Core execution methods
    void run();
    void runParallel();

    // Utility methods
    std::vector<Node::Ptr> findLeafNodes() const;
    std::vector<Node::Ptr> topologicalSort();
    std::vector<std::vector<Node::Ptr>> groupNodesByLevel() const;
    void executeNodesInParallel(const std::vector<Node::Ptr> &nodes);

    // Core data members
    std::vector<Node::Ptr> m_nodes;
    std::vector<Node::Ptr> m_execution_order;
    bool m_compiled = false;
    bool m_parallel_enabled = false;
    size_t m_multicore_threshold = 3;                  // Minimum tasks to use multicore
    std::shared_ptr<InputNode> m_input_node = nullptr; // Automatic input node

    // Pipeline state management
    bool m_is_branched = false;
};

} // namespace mainera

#endif // GRAPH_HPP