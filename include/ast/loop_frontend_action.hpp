#pragma once

#include "ast/loop_ast_consumer.hpp"
#include "core/visibility.hpp"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include <memory>

class ACCELERA_API LoopFrontendAction : public clang::ASTFrontendAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef file) override;

  void EndSourceFileAction() override;

private:
  LoopASTConsumer *Consumer = nullptr;
};
