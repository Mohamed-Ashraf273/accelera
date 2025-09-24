#include "nodes/model.hpp"
#include "nodes/input.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <filesystem>
#include <chrono>
#include <iostream>
using namespace std;

namespace py = pybind11;

namespace mainera {

ModelNode::ModelNode(const std::string &name, py::object py_func)
    : Node(NodeType::MODEL, name, py_func) {

  // Validate that this is a model object with fit method
  if (!py::hasattr(py_func, "fit")) {
    throw std::runtime_error("Model node '" + name +
                             "' must have a 'fit' method");
  }
}

void ModelNode::execute() {
  try {
    std::shared_ptr<InputNode> input = getInput();

    if (!input) {
      throw std::runtime_error("Model node '" + name +
                               "' requires a valid input");
    }

    py::object X = input->getX();
    py::object y = input->getY();

    if (X.is_none() || y.is_none()) {
      throw std::runtime_error("Model node '" + name +
                               "' received None inputs (X or y)");
    }

    if (!py::hasattr(X, "shape") || !py::hasattr(y, "shape")) {
      throw std::runtime_error("Model node '" + name +
                               "' inputs must be array-like objects");
    }

    py::object model_instance =
        py::module::import("copy").attr("deepcopy")(py_func);
    
    namespace fs = std::filesystem;
    py::module_ joblib = py::module_::import("joblib");
    py::module_ np = py::module_::import("numpy");
    
    py::object hash_obj = joblib.attr("hash")(py::make_tuple(model_instance, X, y));
    std::string hash_value = hash_obj.cast<std::string>();
    std::string currentPath = fs::current_path().string();
    fs::create_directory(currentPath+"\\cache");
    std::string model_path = currentPath + "\\cache\\" + hash_value + ".pkl";
    
    if (should_create_new_data) {
      py::object X_copy = X.attr("copy")();
      py::object y_copy = y.attr("copy")();
      auto new_input = std::make_shared<InputNode>();
      if(fs::exists(model_path)){
        py::object fitted = joblib.attr("load")(model_path);
        new_input->setFittedModel(fitted);
      }else{
        try {
          model_instance.attr("fit")(X_copy, y_copy);
        } catch (const py::error_already_set &e) {
          throw std::runtime_error("Python error in model fitting: " +
                                   std::string(e.what()));
        }
        new_input->setFittedModel(model_instance);
        cout<< "saving model to " << model_path << '\n';
        joblib.attr("dump")(model_instance, model_path);
      }
      new_input->setInputData(X_copy, y_copy);
      setOutput(new_input);
    } else {
      if(fs::exists(model_path)){
        py::object fitted = joblib.attr("load")(model_path);
        input->setFittedModel(fitted);
      }else{
        try {
          model_instance.attr("fit")(X, y);
        } catch (const py::error_already_set &e) {
          throw std::runtime_error("Python error in model fitting: " +
                                   std::string(e.what()));
        }
        input->setFittedModel(model_instance);
        cout<< "saving model to " << model_path;
        joblib.attr("dump")(model_instance, model_path);
      }
      setOutput(input);
    }

  } catch (const py::error_already_set &e) {
    throw std::runtime_error("ModelNode: Python error during execution: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error("ModelNode: Error during execution: " +
                             std::string(e.what()));
  }
}

} // namespace mainera