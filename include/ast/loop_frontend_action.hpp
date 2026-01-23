#pragma once

#include "ast/loop_ast_consumer.hpp"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include <memory>

class LoopFrontendAction : public clang::ASTFrontendAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef file) override;

  void EndSourceFileAction() override;

private:
  LoopASTConsumer *Consumer = nullptr;
};
