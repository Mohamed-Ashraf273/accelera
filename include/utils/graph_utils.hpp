#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include <string>

namespace mainera {

class Graph; // Forward declaration

// Serialize a graph to XML format
void serialize_graph(const Graph &graph, const std::string &filepath);

} // namespace mainera

#endif // GRAPH_UTILS_HPP
