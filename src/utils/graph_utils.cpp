#include <fstream>
#include <map>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>
#include <string>

#include "core/graph.hpp"
#include "utils/graph_utils.hpp"

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
    case NodeType::METRIC:
      layer_type = "METRIC";
      break;
    default:
      layer_type = "UNKNOWN";
      break;
    }

    file << "\t\t<layer id=\"" << i << "\" name=\"" << node->name
         << "\" type=\"" << layer_type << "\" version=\"opset1\">\n";
    file << "\t\t\t<data/>\n";

    // Input ports (only if node has a source)
    if (node->type == NodeType::MERGE) {
      file << "\t\t\t<input>\n";
      for (int i = 0; i < node->getSourceNodes().size(); i++) {
        file << "\t\t\t\t<port id=\"" << i << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</input>\n";
    } else {
      if (node->getSourceNode()) {
        file << "\t\t\t<input>\n";
        file << "\t\t\t\t<port id=\"0\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t\t<dim>-1</dim>\n";
        file << "\t\t\t\t</port>\n";
        file << "\t\t\t</input>\n";
      }
    }

    // Output ports (count how many nodes use this node as source)
    int num_outputs = 0;
    for (const auto &other_node : m_nodes) {
      // Count regular single source connections
      if (other_node->getSourceNode() == node) {
        num_outputs++;
      }
      // Count merge node multiple source connections
      if (other_node->type == NodeType::MERGE) {
        const auto &source_nodes = other_node->getSourceNodes();
        for (const auto &source : source_nodes) {
          if (source == node) {
            num_outputs++;
          }
        }
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

    // Handle merge nodes with multiple source nodes
    if (target_node->type == NodeType::MERGE) {
      const auto &source_nodes = target_node->getSourceNodes();
      for (size_t port_idx = 0; port_idx < source_nodes.size(); ++port_idx) {
        const auto &source_node = source_nodes[port_idx];
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
            // Also check if other merge nodes use this as a source
            if (other_node->type == NodeType::MERGE) {
              const auto &other_sources = other_node->getSourceNodes();
              for (const auto &other_source : other_sources) {
                if (other_source == source_node &&
                    other_node.get() < target_node.get()) {
                  source_output_port++;
                }
              }
            }
          }

          file << "\t\t<edge from-layer=\"" << source_id << "\" from-port=\""
               << source_output_port << "\" to-layer=\"" << i << "\" to-port=\""
               << port_idx << "\"/>\n";
        }
      }
    } else {
      // Handle regular nodes with single source node
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
            // Also check if merge nodes use this as a source
            if (other_node->type == NodeType::MERGE) {
              const auto &other_sources = other_node->getSourceNodes();
              for (const auto &other_source : other_sources) {
                if (other_source == source_node &&
                    other_node.get() < target_node.get()) {
                  source_output_port++;
                }
              }
            }
          }

          file << "\t\t<edge from-layer=\"" << source_id << "\" from-port=\""
               << source_output_port << "\" to-layer=\"" << i
               << "\" to-port=\"0\"/>\n";
        }
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

std::string nodeTypeToString(NodeType type) {
  switch (type) {
  case NodeType::INPUT:
    return "INPUT";
  case NodeType::PREPROCESS:
    return "PREPROCESS";
  case NodeType::FEATURE:
    return "FEATURE";
  case NodeType::MODEL:
    return "MODEL";
  case NodeType::PREDICT:
    return "PREDICT";
  case NodeType::MERGE:
    return "MERGE";
  case NodeType::METRIC:
    return "METRIC";
  default:
    return "UNKNOWN";
  }
}

bool validateNodeConnection(Node::Ptr newNode, Node::Ptr sourceNode) {
  switch (newNode->type) {
  case NodeType::PREPROCESS:
    return sourceNode->type == NodeType::INPUT ||
           sourceNode->type == NodeType::PREPROCESS;

  case NodeType::MODEL:
    return sourceNode->type == NodeType::INPUT ||
           sourceNode->type == NodeType::PREPROCESS;

  case NodeType::PREDICT:
    return sourceNode->type == NodeType::MODEL;

  case NodeType::METRIC:
    return sourceNode->type == NodeType::PREDICT ||
           sourceNode->type == NodeType::MERGE;

  case NodeType::MERGE:
    return sourceNode->type == NodeType::PREDICT;

  case NodeType::FEATURE:
    return true;

  case NodeType::INPUT:
    return sourceNode == nullptr;

  default:
    return false;
  }
}
} // namespace mainera