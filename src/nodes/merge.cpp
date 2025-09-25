#include <iostream>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "nodes/input.hpp"
#include "nodes/merge.hpp"

namespace py = pybind11;

namespace mainera {

MergeNode::MergeNode(const std::string &name, py::object py_func)
    : Node(NodeType::MERGE, name, py_func) {}

py::object hardVoting(const py::list &predictions) {
  if (predictions.size() == 0) {
    throw std::runtime_error("No predictions for hard voting");
  }

  try {
    // Use numpy from Python to handle the voting
    py::module_ np = py::module_::import("numpy");

    std::cout << "Hard voting with " << predictions.size() << " models"
              << std::endl;

    // Convert predictions to numpy arrays
    py::list np_predictions = py::list();
    for (const auto &pred : predictions) {
      py::object np_pred = np.attr("array")(pred);
      np_predictions.append(np_pred);
    }

    // Stack predictions to shape (n_models, n_samples, ...)
    py::object stacked = np.attr("array")(np_predictions);

    // Get the first prediction to determine shape
    py::object first_pred = np_predictions[0];
    py::object ndim = first_pred.attr("ndim");
    int dimensions = ndim.cast<int>();

    if (dimensions == 1) {
      // Direct class predictions - use scipy.stats.mode
      py::module_ scipy_stats = py::module_::import("scipy.stats");
      try {
        // Try newer scipy API first
        py::object mode_result = scipy_stats.attr("mode")(
            stacked, py::arg("axis") = 0, py::arg("keepdims") = false);
        return mode_result.attr("mode");
      } catch (const py::error_already_set &) {
        // Fallback to older scipy API
        py::object mode_result =
            scipy_stats.attr("mode")(stacked, py::arg("axis") = 0);
        return mode_result[0]; // mode_result is a tuple (mode, count)
      }
    } else {
      // Probability predictions - take argmax for each model, then vote
      py::object class_preds = np.attr("argmax")(stacked, py::arg("axis") = 2);
      py::module_ scipy_stats = py::module_::import("scipy.stats");
      try {
        // Try newer scipy API first
        py::object mode_result = scipy_stats.attr("mode")(
            class_preds, py::arg("axis") = 0, py::arg("keepdims") = false);
        return mode_result.attr("mode");
      } catch (const py::error_already_set &) {
        // Fallback to older scipy API
        py::object mode_result =
            scipy_stats.attr("mode")(class_preds, py::arg("axis") = 0);
        return mode_result[0]; // mode_result is a tuple (mode, count)
      }
    }

  } catch (const py::error_already_set &e) {
    std::cerr << "Python error in hardVoting: " << e.what() << std::endl;
    throw std::runtime_error("Python error in hard voting: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    std::cerr << "Error in hardVoting: " << e.what() << std::endl;
    throw;
  }
}

void MergeNode::execute() {
  try {
    // Collect results from all source nodes (branches)
    py::list branch_results = py::list();

    // Get all source nodes (these should be the terminal nodes of each branch)
    auto sources = getSourceNodes();

    if (sources.empty()) {
      throw std::runtime_error("MergeNode has no source nodes to merge");
    }

    // Collect results from each branch
    for (auto &source : sources) {
      if (source) {
        // Get the result from each source node
        py::object source_result = source->getData();
        branch_results.append(source_result);
      }
    }

    // Get the merge strategy string
    if (py_func.is_none()) {
      throw std::runtime_error("Merge strategy is not set");
    }

    // Cast py_func to string to get the strategy
    std::string strategy;
    if (py::isinstance<py::str>(py_func)) {
      strategy = py_func.cast<std::string>();
    } else {
      strategy = "hard_voting"; // Default fallback
    }

    std::cout << "Applying merge strategy: " << strategy << " with "
              << branch_results.size() << " models" << std::endl;

    // Apply the specified merge strategy
    if (strategy == "hard_voting" || strategy == "majority_voting") {
      setData(hardVoting(branch_results));
    } else {
      throw std::runtime_error(
          "Only 'hard_voting' strategy is currently supported. Got: " +
          strategy);
    }

  } catch (const std::exception &e) {
    std::cerr << "Error in MergeNode::execute(): " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "Unknown error in MergeNode::execute()" << std::endl;
    throw std::runtime_error("Unknown error in merge execution");
  }
}

} // namespace mainera
