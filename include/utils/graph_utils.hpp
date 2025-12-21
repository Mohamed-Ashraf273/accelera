#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include "core/node.hpp"
#include "core/visibility.hpp"

#include <string>

namespace accelera {

class Graph; // Forward declaration

// Serialize a graph to XML format
ACCELERA_API void serialize_graph(const Graph &graph,
                                  const std::string &filepath);
ACCELERA_API void log_warning(const std::string &message);
ACCELERA_API bool validateNodeConnection(Node::Ptr newNode,
                                         Node::Ptr sourceNode);
ACCELERA_API std::string nodeTypeToString(NodeType type);
ACCELERA_API void saveAsCsv(const std::string &directory, py::object X,
                            py::object y, std::string name);

} // namespace accelera

#endif // GRAPH_UTILS_HPP
