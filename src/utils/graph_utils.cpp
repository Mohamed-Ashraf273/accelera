#include "core/graph.hpp"
#include "core/node.hpp"
#include <fstream>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace aistudio {

void serialize_graph(const Graph &graph, const std::string &filepath) {
  const auto &m_nodes = graph.getNodes();

  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  // Write OpenVINO-compatible XML format
  file << "<?xml version=\"1.0\"?>\n";
  file << "<net name=\"AI_Studio_Pipeline\" version=\"11\">\n";

  // Add layers (nodes)
  file << "\t<layers>\n";
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];

    // Use the actual node type names
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
         << "\" type=\"" << layer_type << "\">\n";

    // Add data section for parameters
    if (!node->py_func.is_none() && py::hasattr(node->py_func, "get_params")) {
      try {
        py::dict params = node->py_func.attr("get_params")();
        file << "\t\t\t<data";
        for (auto item : params) {
          std::string key = py::str(item.first).cast<std::string>();
          std::string value = py::str(item.second).cast<std::string>();
          // Clean up value for XML
          if (value != "None") {
            file << " " << key << "=\"" << value << "\"";
          }
        }
        file << "/>\n";
      } catch (...) {
        file << "\t\t\t<data/>\n";
      }
    } else {
      file << "\t\t\t<data/>\n";
    }

    // Add input ports
    if (node->getInputEdges().size() > 0) {
      file << "\t\t\t<input>\n";
      for (size_t in_idx = 0; in_idx < node->getInputEdges().size(); ++in_idx) {
        file << "\t\t\t\t<port id=\"" << in_idx << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>1</dim>\n";  // Batch size
        file << "\t\t\t\t\t<dim>-1</dim>\n"; // Dynamic dimension
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</input>\n";
    }

    // Add output ports
    if (node->getOutputEdges().size() > 0) {
      file << "\t\t\t<output>\n";
      for (size_t out_idx = 0; out_idx < node->getOutputEdges().size();
           ++out_idx) {
        file << "\t\t\t\t<port id=\""
             << (node->getInputEdges().size() + out_idx)
             << "\" precision=\"FP32\">\n";
        file << "\t\t\t\t\t<dim>1</dim>\n";  // Batch size
        file << "\t\t\t\t\t<dim>-1</dim>\n"; // Dynamic dimension
        file << "\t\t\t\t</port>\n";
      }
      file << "\t\t\t</output>\n";
    }

    file << "\t\t</layer>\n";
  }
  file << "\t</layers>\n";

  // Add edges (connections)
  file << "\t<edges>\n";
  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &node = m_nodes[i];
    const auto &outputs = node->getOutputEdges();

    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
      const auto &edge = outputs[out_idx];
      if (!edge)
        continue;

      // Find which node this edge connects to
      for (size_t j = 0; j < m_nodes.size(); ++j) {
        if (i == j)
          continue; // Skip self
        const auto &target_node = m_nodes[j];
        const auto &inputs = target_node->getInputEdges();

        for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
          if (inputs[in_idx] == edge) {
            // Calculate port IDs according to OpenVINO convention
            size_t from_port = node->getInputEdges().size() + out_idx;
            size_t to_port = in_idx;

            file << "\t\t<edge from-layer=\"" << i << "\" from-port=\""
                 << from_port << "\" to-layer=\"" << j << "\" to-port=\""
                 << to_port << "\"/>\n";
          }
        }
      }
    }
  }
  file << "\t</edges>\n";

  // Add meta information
  file << "\t<meta_data>\n";
  file << "\t\t<MO_version value=\"AI Studio Graph Serializer\"/>\n";
  file << "\t\t<cli_parameters>\n";
  file << "\t\t\t<input_shape value=\"[1,-1]\"/>\n";
  file << "\t\t\t<transform value=\"\"/>\n";
  file << "\t\t</cli_parameters>\n";
  file << "\t</meta_data>\n";

  file << "</net>\n";

  file.close();
}

} // namespace aistudio