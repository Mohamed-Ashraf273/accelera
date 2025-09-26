#ifndef PREPROCESS_NODE_HPP
#define PREPROCESS_NODE_HPP

#include "core/node.hpp"

namespace mainera
{

class MAINERA_API PreprocessNode : public Node
{
   public:
    PreprocessNode(const std::string& name, py::object py_func);
    void execute() override;
    void setData(std::shared_ptr<InputNode> input);
    std::shared_ptr<InputNode> getData() const;

   private:
    std::shared_ptr<InputNode> m_input;
};

}  // namespace mainera

#endif  // PREPROCESS_NODE_HPP