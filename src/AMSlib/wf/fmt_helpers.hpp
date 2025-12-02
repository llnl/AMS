#include <fmt/format.h>

#include <filesystem>

template <>
struct fmt::formatter<std::filesystem::path>
    : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const std::filesystem::path& p, FormatContext& ctx) const
  {
    return fmt::formatter<std::string_view>::format(p.string(), ctx);
  }
};
