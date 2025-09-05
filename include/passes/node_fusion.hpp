#pragma once

#include "core/graph.hpp"

namespace mainera {
namespace passes {

class NodeFusion {
public:
  static void apply(Graph &graph);
};

} // namespace passes
} // namespace mainera
