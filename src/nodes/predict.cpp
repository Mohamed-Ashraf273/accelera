#include "nodes/predict.hpp"
#include "core/graph.hpp"
#include "nodes/input.hpp"
#include <functional>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace mainera {

PredictNode::PredictNode(const std::string &name, py::object py_func)
    : Node(NodeType::PREDICT, name, py_func) {}

void PredictNode::execute() {
  try {
    std::shared_ptr<InputNode> input = getInput();

    if (!input) {
      throw std::runtime_error("Predict node '" + name +
                               "' requires a valid input");
    }

    py::object fitted_model = input->getFittedModel();

    py::object test_data = py_func;
    if (test_data.is_none()) {
      throw std::runtime_error("PredictNode: No test data provided");
    }

    py::object preprocessed_test_data = test_data;
    if (getGraph()) {
      const auto preprocess_functions =
          getGraph()->getPreprocessingFunctions(shared_from_this());

      // Apply each preprocessing function in order
      for (const auto &preprocess_func : preprocess_functions) {
        if (!preprocess_func.is_none()) {
          try {
            preprocessed_test_data = preprocess_func(preprocessed_test_data);
          } catch (const py::error_already_set &e) {
            throw std::runtime_error(
                "Error applying preprocessing to test data: " +
                std::string(e.what()));
          }
        }
      }
    }

    if (fitted_model.is_none()) {
      throw std::runtime_error("PredictNode: No fitted model found in input");
    }

    // Validate input dimensions if it's a numpy array
    if (py::isinstance<py::array>(preprocessed_test_data)) {
      py::array test_array = preprocessed_test_data.cast<py::array>();
      if (test_array.ndim() < 1) {
        throw std::runtime_error(
            "PredictNode: Test data must be at least 1-dimensional");
      }
    }

    py::object predictions;
    try {
      predictions = fitted_model.attr("predict")(preprocessed_test_data);
    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in prediction: " +
                               std::string(e.what()));
    }

    auto new_input = std::make_shared<InputNode>();
    new_input->setInputData(preprocessed_test_data, predictions);
    new_input->setFittedModel(fitted_model);
    setOutput(new_input);

  } catch (const std::exception &e) {
    throw std::runtime_error("Error in PredictNode::execute(): " +
                             std::string(e.what()));
  }
}

} // namespace mainera