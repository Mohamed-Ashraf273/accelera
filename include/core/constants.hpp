#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// Backward-compatible wrapper.
// Prefer including `core/config.hpp` and using `accelera::config::*`.

#include "core/config.hpp"

namespace accelera {
inline constexpr auto CACHE_DIR = config::kCacheDirName;
inline constexpr auto CONFIG_FILE = config::kConfigFileName;
} // namespace accelera

#endif // CONSTANTS_HPP
