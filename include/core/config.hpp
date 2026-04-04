#ifndef ACCELERA_CORE_CONFIG_HPP
#define ACCELERA_CORE_CONFIG_HPP

#include <string_view>

namespace accelera {
namespace config {

inline constexpr std::string_view kCacheDirName = ".accelera_cache";
inline constexpr std::string_view kConfigFileName = ".accelera_config";

} // namespace config
} // namespace accelera

#endif // ACCELERA_CORE_CONFIG_HPP
