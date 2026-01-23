#include "ast/loop_ast_consumer.hpp"

using namespace clang;

LoopASTConsumer::LoopASTConsumer(ASTContext *Context) : Visitor(Context) {}

void LoopASTConsumer::HandleTranslationUnit(ASTContext &Context) {
  Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  loops = Visitor.getLoops();
}

const std::vector<accelera::LoopInfo> &LoopASTConsumer::getLoops() const {
  return loops;
}
