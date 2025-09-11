#pragma once

#include "core/graph.hpp"

namespace mainera {
namespace passes {

class MAINERA_API NodeFusion {
public:
  static void apply(Graph &graph);
};

} // namespace passes
} // namespace mainera
