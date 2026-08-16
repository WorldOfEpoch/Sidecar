#pragma once

#include "sidecar/hardware/hardware.hpp"

#include <string>

namespace sidecar::hardware {

[[nodiscard]] std::string DiscoveryReportToJson(const DiscoveryReport& report);
[[nodiscard]] std::string TopologyToJson(const DiscoveryReport& report);
[[nodiscard]] std::string FormatSystemInformation(const DiscoveryReport& report);
[[nodiscard]] std::string FormatTopology(const DiscoveryReport& report);

}  // namespace sidecar::hardware

