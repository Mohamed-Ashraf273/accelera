#include "core/graph.hpp"
#include "core/node.hpp"
#include <fstream>
#include <map>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace mainera {

void serialize_graph(const Graph &graph, const std::string &filepath) {
  const auto &m_nodes = graph.getNodes();

  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  file << "<?xml version=\"1.0\"?>\n";
  file << "<net name=\"AI_Studio_Pipeline\" version=\"10\">\n";

  file << "\t<layers>\n";
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];

    std::string layer_type;
    switch (node->type) {
    case NodeType::INPUT:
      layer_type = "INPUT";
      break;
    case NodeType::PREPROCESS:
      layer_type = "PREPROCESS";
      break;
    case NodeType::FEATURE:
      layer_type = "FEATURE";
      break;
    case NodeType::MODEL:
      layer_type = "MODEL";
      break;
    case NodeType::PREDICT:
      layer_type = "PREDICT";
      break;
    case NodeType::MERGE:
      layer_type = "MERGE";
      break;
    default:
      layer_type = "UNKNOWN";
      break;
    }

    file << "\t\t<layer id=\"" << i << "\" name=\"" << node->name
         << "\" type=\"" << layer_type << "\" version=\"opset1\">\n";
    file << "\t\t\t<data/>\n";

    // Input ports (start from 0)
    if (node->getInputEdge()) {
      file << "\t\t\t<input>\n";
      file << "\t\t\t\t<port id=\"0\" precision=\"FP32\">\n";
      file << "\t\t\t\t\t<dim>1</dim>\n";
      file << "\t\t\t\t\t<dim>-1</dim>\n";
      file << "\t\t\t\t</port>\n";
      file << "\t\t\t</input>\n";
    }

    // Output ports (start from next available port number)
    if (node->getOutputEdge()) {
      // Count connections to determine how many output ports needed
      int num_outputs = 0;
      for (size_t j = 0; j < m_nodes.size(); ++j) {
        if (i == j)
          continue;
        const auto &target_node = m_nodes[j];
        const auto &input_edge = target_node->getInputEdge();
        if (input_edge && input_edge.get() == node->getOutputEdge().get()) {
          num_outputs++;
        }
      }

      file << "\t\t\t<output>\n";

      // For INPUT nodes, start output ports from 0
      // For other nodes with input, start output ports from 1
      int start_port = (node->getInputEdge()) ? 1 : 0;

      // Create only the exact number of output ports needed
      for (int port = 0; port < std::max(1, num_outputs); ++port) {
        file << "\t\t\t\t<port id=\"" << (start_port + port)
             << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>1</dim>\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</output>\n";
    }

    file << "\t\t</layer>\n";
  }
  file << "\t</layers>\n";

  file << "\t<edges>\n";

  // Build connections
  std::map<size_t, std::vector<size_t>> node_connections;
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &source_node = m_nodes[i];
    const auto &source_output = source_node->getOutputEdge();

    if (source_output) {
      for (size_t j = 0; j < m_nodes.size(); ++j) {
        if (i == j)
          continue;

        const auto &target_node = m_nodes[j];
        const auto &target_input = target_node->getInputEdge();

        if (target_input && target_input.get() == source_output.get()) {
          node_connections[i].push_back(j);
        }
      }
    }
  }

  // Generate edges with correct port numbering
  for (const auto &conn : node_connections) {
    size_t from_node = conn.first;
    const auto &targets = conn.second;

    // Determine starting port for source node
    int source_start_port = (m_nodes[from_node]->getInputEdge()) ? 1 : 0;

    for (size_t idx = 0; idx < targets.size(); ++idx) {
      size_t to_node = targets[idx];

      int from_port = source_start_port + idx;
      int to_port = 0; // Target always uses port 0 for input

      file << "\t\t<edge from-layer=\"" << from_node << "\" from-port=\""
           << from_port << "\" " << "to-layer=\"" << to_node << "\" to-port=\""
           << to_port << "\"/>\n";
    }
  }

  file << "\t</edges>\n";
  file << "</net>\n";

  file.close();
}

} // namespace mainera