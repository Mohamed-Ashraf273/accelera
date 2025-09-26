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
MAINERA_API bool validateNodeConnection(Node::Ptr newNode,
                                        Node::Ptr sourceNode);
MAINERA_API std::string nodeTypeToString(NodeType type);

} // namespace mainera

#endif // GRAPH_UTILS_HPP
