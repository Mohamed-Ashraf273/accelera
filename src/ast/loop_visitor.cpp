#include "ast/loop_visitor.hpp"

#include "clang/Lex/Lexer.h"

using namespace clang;

LoopVisitor::LoopVisitor(ASTContext *Context) : Context(Context) {}

bool LoopVisitor::VisitForStmt(ForStmt *FS) {
  extractLoop(FS, "for");
  return true;
}

bool LoopVisitor::VisitWhileStmt(WhileStmt *WS) {
  extractLoop(WS, "while");
  return true;
}

const std::vector<accelera::LoopInfo> &LoopVisitor::getLoops() const {
  return loops;
}

void LoopVisitor::extractLoop(Stmt *S, const std::string &type) {
  SourceManager &SM = Context->getSourceManager();

  if (!SM.isWrittenInMainFile(S->getBeginLoc()))
    return;

  accelera::LoopInfo info;
  info.type = type;
  info.start_line = SM.getSpellingLineNumber(S->getBeginLoc());
  info.end_line = SM.getSpellingLineNumber(S->getEndLoc());

  CharSourceRange range = CharSourceRange::getTokenRange(S->getSourceRange());

  info.code = Lexer::getSourceText(range, SM, Context->getLangOpts()).str();

  loops.push_back(info);
}
