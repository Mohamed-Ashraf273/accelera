#include "ast/loop_frontend_action.hpp"

using namespace clang;

std::unique_ptr<ASTConsumer>
LoopFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                      llvm::StringRef file) {
  auto consumer = std::make_unique<LoopASTConsumer>(&CI.getASTContext());
  Consumer = consumer.get();
  return consumer;
}

void LoopFrontendAction::EndSourceFileAction() {
  if (Consumer) {
    const auto &loops = Consumer->getLoops();
  }
}
