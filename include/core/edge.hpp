#pragma once
#include <memory>
#include <string>

namespace aistudio {

class Node; // forward declaration

class Edge {
public:
    using Ptr = std::shared_ptr<Edge>;

    std::shared_ptr<Node> from;
    std::shared_ptr<Node> to;
    std::string label;

    Edge(std::shared_ptr<Node> f, std::shared_ptr<Node> t, const std::string& l = "")
        : from(std::move(f)), to(std::move(t)), label(l) {}
};

} // namespace aistudio
