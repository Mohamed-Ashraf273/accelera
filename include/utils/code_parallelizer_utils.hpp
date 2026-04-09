#ifndef CODE_PARALLELIZER_UTILS_HPP
#define CODE_PARALLELIZER_UTILS_HPP

#include "core/visibility.hpp"

#include <string>
#include <vector>

namespace accelera {

struct LoopInfo {
  std::string type;
  unsigned start_line;
  unsigned end_line;
  std::string code;
};

ACCELERA_API std::vector<LoopInfo>
extract_loops(const std::string &code,
              const std::vector<std::string> &clang_args = {});

ACCELERA_API bool write_loops_to_json(const std::vector<LoopInfo> &loops,
                                      const std::string &output_file);

} // namespace accelera

#endif
