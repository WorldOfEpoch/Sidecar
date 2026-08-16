#pragma once

#include "sidecar/hardware/hardware.hpp"

#include <string>
#include <vector>

namespace sidecar::hardware {

void DiscoverCudaDevices(MachineSnapshot& snapshot, std::vector<std::string>& warnings);
void EnrichNvidiaDevicesWithNvml(MachineSnapshot& snapshot,
                                 std::vector<std::string>& warnings);

}  // namespace sidecar::hardware

