#include "core/edge.hpp"

namespace mainera {

Edge::Edge() : m_isReady(false) {}

Edge::~Edge() {}

py::object Edge::getData() { return m_data; }

void Edge::setData(py::object data) {
  m_data = data;
  m_isReady = !data.is_none();
}

bool Edge::isReady() const { return m_isReady; }

void Edge::setReady(bool status) { m_isReady = status; }

} // namespace mainera