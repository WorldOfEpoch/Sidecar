#pragma once

#include <string_view>

namespace sidecar {

struct VersionInfo {
    std::string_view version;
    std::string_view spec_version;
    std::string_view git_commit;
};

[[nodiscard]] VersionInfo CurrentVersionInfo() noexcept;

}  // namespace sidecar

