#include <filesystem>
#include <fstream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>
#include <string>

#include "core/graph.hpp"
#include "utils/graph_utils.hpp"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace accelera {

void serialize_graph(const Graph &graph, const std::string &filepath) {
  const auto &m_nodes = graph.getNodes();

  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  file << "<?xml version=\"1.0\"?>\n";
  file << "<net name=\"Accelera_Pipeline\" version=\"10\">\n";

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
         << "\" type=\"" << layer_type << "\" version=\"opset1\""
         << " selected_in_path=\""
         << (node->selected_in_path ? "true" : "false") << "\">\n";
    file << "\t\t\t<data/>\n";

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

    int num_outputs = 0;
    for (const auto &other_node : m_nodes) {
      if (other_node->getSourceNode() == node) {
        num_outputs++;
      }
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

  for (size_t i = 0; i < m_nodes.size(); ++i) {
    const auto &target_node = m_nodes[i];

    if (target_node->type == NodeType::MERGE) {
      const auto &source_nodes = target_node->getSourceNodes();
      for (size_t port_idx = 0; port_idx < source_nodes.size(); ++port_idx) {
        const auto &source_node = source_nodes[port_idx];
        auto source_it = node_to_id.find(source_node.get());
        if (source_it != node_to_id.end()) {
          size_t source_id = source_it->second;

          int source_output_port = 0;
          for (const auto &other_node : m_nodes) {
            if (other_node->getSourceNode() == source_node) {
              if (other_node == target_node) {
                break;
              }
              source_output_port++;
            }
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
      const auto &source_node = target_node->getSourceNode();
      if (source_node) {
        auto source_it = node_to_id.find(source_node.get());
        if (source_it != node_to_id.end()) {
          size_t source_id = source_it->second;

          int source_output_port = 0;
          for (const auto &other_node : m_nodes) {
            if (other_node->getSourceNode() == source_node) {
              if (other_node == target_node) {
                break;
              }
              source_output_port++;
            }
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
  bool valid = false;
  switch (newNode->type) {
  case NodeType::PREPROCESS:
    valid = sourceNode->type == NodeType::INPUT ||
            sourceNode->type == NodeType::PREPROCESS;
    break;

  case NodeType::MODEL:
    valid = sourceNode->type == NodeType::INPUT ||
            sourceNode->type == NodeType::PREPROCESS;
    break;

  case NodeType::PREDICT:
    valid = sourceNode->type == NodeType::MODEL;
    break;

  case NodeType::METRIC:
    valid = sourceNode->type == NodeType::PREDICT ||
            sourceNode->type == NodeType::MERGE;
    break;

  case NodeType::MERGE:
    valid = sourceNode->type == NodeType::PREDICT;
    break;

  case NodeType::INPUT:
    valid = sourceNode == nullptr;
    break;
  }
  if (!valid)
    return false;
  return true;
}

void saveAsCsv(const std::string &directory, py::object X, py::object y,
               std::string name) {
  try {
    fs::path dirPath(directory);
    if (!fs::exists(dirPath)) {
      fs::create_directories(dirPath);
    }

    fs::path csvPath = dirPath / (name + ".csv");
    std::ofstream csvFile(csvPath.string());

    if (!csvFile.is_open()) {
      throw std::runtime_error("Failed to create CSV file: " +
                               csvPath.string());
    }

    py::array_t<double> X_array = py::cast<py::array_t<double>>(X);
    auto X_buf = X_array.request();

    size_t n_samples = X_buf.shape[0];
    size_t n_features = (X_buf.ndim > 1) ? X_buf.shape[1] : 1;

    for (size_t j = 0; j < n_features; ++j) {
      csvFile << "feature_" << j;
      if (j < n_features - 1)
        csvFile << ",";
    }
    if (!y.is_none()) {
      csvFile << ",label";
    }
    csvFile << "\n";

    double *X_ptr = static_cast<double *>(X_buf.ptr);

    py::array_t<double> y_array;
    double *y_ptr = nullptr;
    if (!y.is_none()) {
      y_array = py::cast<py::array_t<double>>(y);
      auto y_buf = y_array.request();
      y_ptr = static_cast<double *>(y_buf.ptr);
    }

    for (size_t i = 0; i < n_samples; ++i) {
      for (size_t j = 0; j < n_features; ++j) {
        size_t index = (X_buf.ndim > 1) ? (i * n_features + j) : i;
        csvFile << X_ptr[index];
        if (j < n_features - 1)
          csvFile << ",";
      }

      if (!y.is_none()) {
        csvFile << "," << y_ptr[i];
      }
      csvFile << "\n";
    }

    csvFile.close();

  } catch (const std::exception &e) {
    throw std::runtime_error("Error saving data to disc from node '" + name +
                             "': " + e.what());
  }
}
} // namespace accelera