#pragma once

#include "core/graph.hpp"

namespace aistudio {
namespace passes {

class NodeFusion {
public:
  static void apply(Graph &graph);
};

} // namespace passes
} // namespace aistudio
