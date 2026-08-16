#pragma once

#include "sidecar/hardware/hardware.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace sidecar::hardware {

[[nodiscard]] std::optional<std::string> NormalizeIdentityValue(
    const std::optional<std::string>& value);
[[nodiscard]] std::optional<std::string> NormalizeSystemUuid(
    const std::optional<std::string>& value);
[[nodiscard]] MachineIdentity BuildMachineIdentity(const MachineIdentityInput& input);

}  // namespace sidecar::hardware

