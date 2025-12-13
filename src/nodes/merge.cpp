#include <pybind11/pybind11.h>
#include <stdexcept>

#include "nodes/input.hpp"
#include "nodes/merge.hpp"

namespace py = pybind11;

namespace accelera {

MergeNode::MergeNode(const std::string &name, py::object py_func)
    : Node(NodeType::MERGE, name, py_func) {}

py::object hardVoting(const py::list &predictions) {
  if (predictions.size() == 0) {
    throw std::runtime_error("No predictions for hard voting");
  }

  try {
    // Use numpy from Python to handle the voting
    py::module_ np = py::module_::import("numpy");

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
    throw std::runtime_error("Python error in hard voting: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw;
  }
}

void MergeNode::execute() {
  try {
    py::list branch_results = py::list();

    // Get all source nodes (these should be the terminal nodes of each branch)
    auto sources = getSourceNodes();

    if (sources.empty()) {
      throw std::runtime_error("MergeNode has no source nodes to merge");
    }

    // Collect results from each branch
    for (auto &source : sources) {
      if (source) {
        std::shared_ptr<py::object> source_result = source->getData();
        if (source_result && !source_result->is_none()) {
          branch_results.append(*source_result);
        }
      }
    }

    if (py_func.is_none()) {
      throw std::runtime_error("Merge strategy is not set");
    }

    std::string strategy;
    if (py::isinstance<py::str>(py_func)) {
      strategy = py_func.cast<std::string>();
    } else {
      strategy = "hard_voting";
    }

    if (strategy == "hard_voting" || strategy == "majority_voting") {
      py::object result = hardVoting(branch_results);
      std::shared_ptr<py::object> result_ptr =
          std::make_shared<py::object>(result);
      setData(result_ptr);
    } else {
      throw std::runtime_error(
          "Only 'hard_voting' strategy is currently supported. Got: " +
          strategy);
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in MergeNode::execute(): " +
                             std::string(e.what()));
  } catch (...) {
    throw std::runtime_error("Unknown error in merge execution");
  }
}

} // namespace accelera
