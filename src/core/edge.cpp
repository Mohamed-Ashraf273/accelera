#include "core/edge.hpp"

namespace mainera {

Edge::Edge() : m_isReady(false) {}

Edge::~Edge() {}

std::shared_ptr<InputNode> Edge::getData() { return m_data; }

void Edge::setData(std::shared_ptr<InputNode> data) {
  m_data = data;
  m_isReady = (data != nullptr);
}

bool Edge::isReady() const { return m_isReady; }

void Edge::setReady(bool status) { m_isReady = status; }

} // namespace mainera