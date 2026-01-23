#pragma once

#ifdef USE_CLANG_AST

#include "core/visibility.hpp"
#include "utils/code_parallelizer_utils.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

#include <string>
#include <vector>

class ACCELERA_API LoopVisitor
    : public clang::RecursiveASTVisitor<LoopVisitor> {
public:
  explicit LoopVisitor(clang::ASTContext *Context);

  bool VisitForStmt(clang::ForStmt *FS);
  bool VisitWhileStmt(clang::WhileStmt *WS);

  const std::vector<accelera::LoopInfo> &getLoops() const;

private:
  clang::ASTContext *Context;
  std::vector<accelera::LoopInfo> loops;

  void extractLoop(clang::Stmt *S, const std::string &type);
};

#endif // USE_CLANG_AST
