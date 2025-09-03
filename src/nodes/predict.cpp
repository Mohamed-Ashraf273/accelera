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
    py::object explicit_test_data = py::none();

    // Check inputs for a fitted model and explicit test data
    for (size_t i = 0; i < inputs.size(); ++i) {
      py::object input = inputs[i];
      if (!input.is_none()) {
        // Check if this input is a fitted scikit-learn model
        if (py::hasattr(input, "fit") && py::hasattr(input, "predict")) {
          // Verify it's fitted by checking for common fitted attributes
          if (py::hasattr(input, "coef_") ||
              py::hasattr(input, "feature_importances_") ||
              py::hasattr(input, "n_features_in_") ||
              py::hasattr(input, "classes_")) {
            fitted_model = input;
          }
        } else {
          // This might be explicit test data (set via set_input)
          explicit_test_data = input;
        }
      }
    }

    if (fitted_model.is_none()) {
      throw std::runtime_error("PredictNode: No fitted model found in inputs");
    }

    // Determine what to use as prediction input
    py::object X_to_predict = py::none();

    // Priority 1: Use test data provided during node construction
    if (!py_func.is_none()) {
      X_to_predict = py_func;
    }
    // Priority 2: Use explicit test data if provided via inputs
    else if (!explicit_test_data.is_none()) {
      X_to_predict = explicit_test_data;
    }
    // Priority 2: Look for input from connected model node (training data
    // fallback)
    else {
      // Get access to the graph's input data that was passed to the pipeline
      // This is a bit tricky - we need to get the original training data
      // For now, let's check if any input edges have training data tuples
      bool found_training_data = false;

      for (const auto &edge : m_inputEdges) {
        if (edge && edge->isReady()) {
          py::object edge_data = edge->getData();

          // If we have a model as input, we can't directly access its training
          // data The model node only outputs the fitted model, not the training
          // data This is a design limitation - we should pass both model and X
          // separately
          if (py::hasattr(edge_data, "fit") &&
              py::hasattr(edge_data, "predict")) {
            continue;
          }
        }
      }

      if (!found_training_data) {
        throw std::runtime_error(
            "PredictNode: No test data provided and no training data "
            "accessible. Please provide test_data to the predict() method.");
      }
    }

    // Final check - if still no prediction input, throw error
    if (X_to_predict.is_none()) {
      throw std::runtime_error(
          "PredictNode: No input data available for prediction");
    }

    // Validate input dimensions if it's a numpy array
    if (py::isinstance<py::array>(X_to_predict)) {
      py::array X_array = X_to_predict.cast<py::array>();
      // Basic shape validation - should be 2D for most models
      if (X_array.ndim() < 1) {
        throw std::runtime_error(
            "PredictNode: Input data must be at least 1-dimensional");
      }
    }

    // BUGFIX: Apply preprocessing to test data using graph's preprocessing
    // functions
    py::object X_preprocessed = X_to_predict;

    // Get preprocessing functions from the graph
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
            // Continue with partially preprocessed data if one function fails
          }
        }
      }
    }

    // Perform prediction using the fitted model on preprocessed test data
    py::object predictions = fitted_model.attr("predict")(X_preprocessed);

    if (py::isinstance<py::array>(predictions)) {
      py::array pred_array = predictions.cast<py::array>();
    }

    // Set the predictions as output
    setOutputs(predictions);

  } catch (const std::exception &e) {
    std::cerr << "Error in PredictNode::execute(): " << e.what() << std::endl;
    throw;
  }
}

} // namespace aistudio