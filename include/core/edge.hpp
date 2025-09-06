#ifndef EDGE_HPP
#define EDGE_HPP

#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mainera {

// Forward declaration to avoid circular dependency
class InputNode;

class __attribute__((visibility("default"))) Edge {
public:
  Edge();
  ~Edge();

  std::shared_ptr<InputNode> getData();
  void setData(std::shared_ptr<InputNode> data);

  bool isReady() const;
  void setReady(bool status);

private:
  std::shared_ptr<InputNode> m_data = nullptr;
  bool m_isReady;
};

} // namespace mainera

#endif // EDGE_HPP