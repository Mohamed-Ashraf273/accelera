#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include "core/node.hpp"
#include "core/visibility.hpp"

#include <string>

namespace mainera {

class Graph; // Forward declaration

// Serialize a graph to XML format
MAINERA_API void serialize_graph(const Graph &graph,
                                 const std::string &filepath);
MAINERA_API void log_warning(const std::string &message);
MAINERA_API void validateNodeConnection(Node::Ptr newNode,
                                        Node::Ptr sourceNode);
MAINERA_API std::string nodeTypeToString(NodeType type);
MAINERA_API void saveAsCsv(const std::string &directory, py::object X,
                           py::object y, std::string name);

} // namespace mainera

#endif // GRAPH_UTILS_HPP
