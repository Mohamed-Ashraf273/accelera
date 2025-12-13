#ifndef PREDICT_NODE_HPP
#define PREDICT_NODE_HPP

#include "core/node.hpp"

namespace accelera {

class ACCELERA_API PredictNode : public Node {
public:
  PredictNode(const std::string &name, py::object py_func);
  void execute() override;
  void setPreprocessedData(std::shared_ptr<py::object> data);
  std::shared_ptr<py::object> getPreprocessedData() const;
  std::vector<py::object> getPreprocessingFunctions() const;

private:
  std::shared_ptr<py::object> m_preprocessed_data = nullptr;
};

} // namespace accelera

#endif // PREDICT_NODE_HPP