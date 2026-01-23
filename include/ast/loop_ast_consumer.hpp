#pragma once

#include "ast/loop_visitor.hpp"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/ASTConsumers.h"

class LoopASTConsumer : public clang::ASTConsumer {
public:
  explicit LoopASTConsumer(clang::ASTContext *Context);

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  const std::vector<accelera::LoopInfo> &getLoops() const;

private:
  LoopVisitor Visitor;
  std::vector<accelera::LoopInfo> loops;
};
