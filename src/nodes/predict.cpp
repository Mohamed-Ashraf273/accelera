#include "nodes/predict.hpp"
#include "core/graph.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace aistudio {

PredictNode::PredictNode(const std::string &name, size_t numInputs,
                         size_t numOutputs, py::object py_func)
    : Node(NodeType::PREDICT, name, numInputs, numOutputs, py_func) {}

void PredictNode::execute() {
  try {
    py::list inputs = collectInputs();

    py::object fitted_model = py::none();

    // Find the fitted model in inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      py::object input = inputs[i];
      if (!input.is_none()) {
        if (py::hasattr(input, "fit") && py::hasattr(input, "predict")) {
          // Verify it's fitted by checking for common fitted attributes
          if (py::hasattr(input, "coef_") ||
              py::hasattr(input, "feature_importances_") ||
              py::hasattr(input, "n_features_in_") ||
              py::hasattr(input, "classes_")) {
            fitted_model = input;
            break;
          }
        }
      }
    }

    if (fitted_model.is_none()) {
      throw std::runtime_error("PredictNode: No fitted model found in inputs");
    }

    // Use test data provided during node construction (py_func)
    py::object X_to_predict = py_func;

    if (X_to_predict.is_none()) {
      throw std::runtime_error(
          "PredictNode: No test data provided to predict node");
    }

    // Validate input dimensions if it's a numpy array
    if (py::isinstance<py::array>(X_to_predict)) {
      py::array X_array = X_to_predict.cast<py::array>();
      if (X_array.ndim() < 1) {
        throw std::runtime_error(
            "PredictNode: Input data must be at least 1-dimensional");
      }
    }

    // Apply preprocessing to test data using graph's preprocessing functions
    py::object X_preprocessed = X_to_predict;

    if (getGraph()) {
      const auto &preprocess_functions =
          getGraph()->getPreprocessingFunctions();

      // Apply each preprocessing function in order
      for (const auto &preprocess_func : preprocess_functions) {
        if (!preprocess_func.is_none()) {
          try {
            X_preprocessed = preprocess_func(X_preprocessed);
          } catch (const py::error_already_set &e) {
            std::cerr << "Warning: Failed to apply preprocessing to test data: "
                      << e.what() << std::endl;
          }
        }
      }
    }

    // Perform prediction using the fitted model on preprocessed test data
    py::object predictions = fitted_model.attr("predict")(X_preprocessed);

    // Set the predictions as output
    setOutputs(predictions);

  } catch (const std::exception &e) {
    std::cerr << "Error in PredictNode::execute(): " << e.what() << std::endl;
    throw;
  }
}

} // namespace aistudio