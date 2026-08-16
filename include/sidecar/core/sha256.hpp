#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace sidecar::core {

[[nodiscard]] std::string Sha256Hex(std::string_view input);
[[nodiscard]] std::string Sha256File(const std::filesystem::path& path);

}  // namespace sidecar::core
