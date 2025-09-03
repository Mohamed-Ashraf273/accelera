#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include <string>

namespace aistudio {

class Graph; // Forward declaration

// Serialize a graph to XML format
void serialize_graph(const Graph &graph, const std::string &filepath);

} // namespace aistudio

#endif // GRAPH_UTILS_HPP
