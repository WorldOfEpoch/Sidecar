#include "sidecar/hardware/hardware.hpp"

#include <stdexcept>

namespace sidecar::hardware {

std::unique_ptr<IHardwareDiscoveryProvider> CreateNativeDiscoveryProvider() {
    throw std::runtime_error("native hardware discovery is not implemented for this operating system");
}

}  // namespace sidecar::hardware

