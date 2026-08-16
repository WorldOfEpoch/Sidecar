#include "sidecar/hardware/native_components.hpp"

namespace sidecar::hardware {

void DiscoverCudaDevices(MachineSnapshot& snapshot, std::vector<std::string>& warnings) {
    snapshot.cuda_available = false;
    warnings.emplace_back("CUDA discovery was disabled at build time");
}

void EnrichNvidiaDevicesWithNvml(MachineSnapshot& snapshot,
                                 std::vector<std::string>& warnings) {
    snapshot.nvml_available = false;
    warnings.emplace_back("NVML discovery was disabled with CUDA support");
}

}  // namespace sidecar::hardware

