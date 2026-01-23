#include "utils/code_parallelizer_utils.hpp"
#include "ast/loop_frontend_action.hpp"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <memory>
#include <sstream>

namespace accelera {

class LoopCollectorAction : public clang::ASTFrontendAction {
public:
  std::shared_ptr<std::vector<LoopInfo>> result_ptr;
  LoopASTConsumer *consumer_ptr = nullptr;

  explicit LoopCollectorAction(std::shared_ptr<std::vector<LoopInfo>> results)
      : result_ptr(results) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef file) override {
    auto consumer = std::make_unique<LoopASTConsumer>(&CI.getASTContext());
    consumer_ptr = consumer.get();
    return consumer;
  }

  void EndSourceFileAction() override {
    if (consumer_ptr && result_ptr) {
      *result_ptr = consumer_ptr->getLoops();
    }
  }
};

std::vector<LoopInfo>
extract_loops(const std::string &filename,
              const std::vector<std::string> &clang_args) {

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open file: " + filename);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string code = buffer.str();
  file.close();

  auto result_ptr = std::make_shared<std::vector<LoopInfo>>();

  try {
    auto action = std::make_unique<LoopCollectorAction>(result_ptr);

    clang::tooling::runToolOnCode(std::move(action), code);

  } catch (const std::bad_alloc &e) {
    throw std::runtime_error("bad_alloc exception: out of memory");
  } catch (const std::exception &e) {
    throw std::runtime_error("Exception during loop extraction: " +
                             std::string(e.what()));
  } catch (...) {
    throw std::runtime_error("Unknown exception during loop extraction");
  }

  return *result_ptr;
}

bool write_loops_to_json(const std::vector<LoopInfo> &loops,
                         const std::string &output_file) {
  std::ofstream out(output_file);
  if (!out.is_open()) {
    throw std::runtime_error("Error: Could not open output file: " +
                             output_file);
  }

  out << "[\n";
  for (size_t i = 0; i < loops.size(); ++i) {
    const auto &loop = loops[i];
    out << "  {\n";
    out << "    \"type\": \"" << loop.type << "\",\n";
    out << "    \"start_line\": " << loop.start_line << ",\n";
    out << "    \"end_line\": " << loop.end_line << ",\n";
    out << "    \"code\": \"";

    for (char c : loop.code) {
      switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
      }
    }

    out << "\"\n";
    out << "  }";
    if (i < loops.size() - 1) {
      out << ",";
    }
    out << "\n";
  }
  out << "]\n";
  out.close();

  return true;
}

} // namespace accelera
