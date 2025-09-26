#include <pybind11/pybind11.h>

#include "utils/graph_utils.hpp"

#include "core/graph.hpp"
#include "core/node.hpp"

#include <fstream>
#include <map>
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

  // Create node ID mapping
  std::map<Node *, size_t> node_to_id;
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    node_to_id[m_nodes[i].get()] = i;
  }

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

    // Input ports (only if node has a source)
    if (node->getSourceNode()) {
      file << "\t\t\t<input>\n";
      file << "\t\t\t\t<port id=\"0\" precision=\"FP32\">\n";
      file << "\t\t\t\t\t<dim>-1</dim>\n";
      file << "\t\t\t\t\t<dim>-1</dim>\n";
      file << "\t\t\t\t</port>\n";
      file << "\t\t\t</input>\n";
    }

    // Output ports (count how many nodes use this node as source)
    int num_outputs = 0;
    for (const auto &other_node : m_nodes) {
      if (other_node->getSourceNode() == node) {
        num_outputs++;
      }
    }

    if (num_outputs > 0) {
      file << "\t\t\t<output>\n";
      for (int port = 0; port < num_outputs; ++port) {
        file << "\t\t\t\t<port id=\"" << port << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</output>\n";
    }

    file << "\t\t</layer>\n";
  }
  file << "\t</layers>\n";

  file << "\t<edges>\n";

  // Build connections based on source node relationships
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &target_node = m_nodes[i];
    const auto &source_node = target_node->getSourceNode();

    if (source_node) {
      // Find source node ID
      auto source_it = node_to_id.find(source_node.get());
      if (source_it != node_to_id.end()) {
        size_t source_id = source_it->second;

        // Count how many outputs the source node has to determine port number
        int source_output_port = 0;
        for (const auto &other_node : m_nodes) {
          if (other_node->getSourceNode() == source_node) {
            if (other_node == target_node) {
              break; // Found our target, use current port
            }
            source_output_port++;
          }
        }

        file << "\t\t<edge from-layer=\"" << source_id << "\" from-port=\""
             << source_output_port << "\" to-layer=\"" << i
             << "\" to-port=\"0\"/>\n";
      }
    }
  }

  file << "\t</edges>\n";
  file << "</net>\n";

  file.close();
}

void log_warning(const std::string &message) {
  std::fprintf(stderr, "WARNING: %s\n", message.c_str());
}
} // namespace mainera