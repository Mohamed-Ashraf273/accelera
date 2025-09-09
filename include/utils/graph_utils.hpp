#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include <string>
#include "core/visibility.hpp"

namespace mainera {

class Graph; // Forward declaration

// Serialize a graph to XML format
MAINERA_API void serialize_graph(const Graph &graph, const std::string &filepath);
MAINERA_API void log_warning(const std::string &message);

} // namespace mainera

#endif // GRAPH_UTILS_HPP
