#include "sidecar/version.hpp"

#include "sidecar/version_config.hpp"

namespace sidecar {

VersionInfo CurrentVersionInfo() noexcept {
    return VersionInfo{
        SIDECAR_VERSION_STRING,
        SIDECAR_SPEC_VERSION_STRING,
        SIDECAR_GIT_COMMIT_STRING,
    };
}

}  // namespace sidecar

