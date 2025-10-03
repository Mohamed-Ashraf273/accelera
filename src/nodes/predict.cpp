#include <functional>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "core/graph.hpp"
#include "nodes/input.hpp"
#include "nodes/model.hpp"
#include "nodes/predict.hpp"

namespace py = pybind11;

namespace mainera {

PredictNode::PredictNode(const std::string &name, py::object py_func)
    : Node(NodeType::PREDICT, name, py_func) {}

void PredictNode::setPreprocessedData(std::shared_ptr<py::object> data) {
  m_preprocessed_data = data;
}

std::shared_ptr<py::object> PredictNode::getPreprocessedData() const {
  return m_preprocessed_data;
}

void PredictNode::execute() {
  try {
    std::shared_ptr<Node> input = getSourceNode();

    if (!input) {
      throw std::runtime_error("Predict node '" + name +
                               "' requires a valid model node");
    }

    py::object fitted_model = *(input->getData());

    if (fitted_model.is_none() || !py::hasattr(fitted_model, "predict")) {
      throw std::runtime_error("Predict node '" + name +
                               "' requires a valid fitted model");
    }

    py::object test_data = py_func["test_data"];
    if (test_data.is_none()) {
      throw std::runtime_error("PredictNode: No test data provided");
    }

    py::object preprocessed_test_data;

    if (getGraph()->getIsExecuted()) {
      preprocessed_test_data = getGraph()->getInputNode()->getX();
    } else {
      preprocessed_test_data = test_data;
    }

    if (getGraph()) {
      const auto preprocess_functions =
          getGraph()->getPreprocessingFunctions(shared_from_this());

      for (const auto &preprocess_func : preprocess_functions) {
        if (!preprocess_func.is_none()) {
          try {
            if (py::hasattr(preprocess_func, "transform")) {
              preprocessed_test_data = py::cast<py::array_t<double>>(
                  preprocess_func.attr("transform")(preprocessed_test_data));
            } else {
              preprocessed_test_data = preprocess_func(preprocessed_test_data);
            }
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

    if (py::isinstance<py::array>(preprocessed_test_data)) {
      py::array test_array = preprocessed_test_data.cast<py::array>();
      if (test_array.ndim() < 1) {
        throw std::runtime_error(
            "PredictNode: Test data must be at least 1-dimensional");
      }
    }

    setPreprocessedData(std::make_shared<py::object>(preprocessed_test_data));

    py::object predictions;
    try {
      py::object output_func = py_func["output_func"];
      int positive_class = py::cast<int>(py_func["positive_class"]);
      if (py::hasattr(fitted_model, output_func)) {
        predictions = fitted_model.attr(output_func)(preprocessed_test_data);
        if (output_func.equal(py::str("predict_proba")) &&
            positive_class != -1) {
          py::array proba_array = predictions.cast<py::array>();
          if (positive_class < 0 || positive_class >= proba_array.shape(1)) {
            throw std::runtime_error(
                "PredictNode: positive_class index is out of bounds");
          }
          py::ssize_t n_rows = py::len(proba_array);
          py::slice all_rows(0, n_rows, 1);
          auto idx = py::make_tuple(all_rows, py::int_(positive_class));
          predictions = proba_array.attr("__getitem__")(idx);
        }
      } else {
        throw std::runtime_error(
            "PredictNode: The model does not have the method '" +
            std::string(py::str(output_func)) + "' as function for prediction");
      }

    } catch (const py::error_already_set &e) {
      throw std::runtime_error("Python error in prediction: " +
                               std::string(e.what()));
    }
    auto predictions_ptr = std::make_shared<py::object>(predictions);
    setData(predictions_ptr);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error in PredictNode::execute(): " +
                             std::string(e.what()));
  }
}

} // namespace mainera