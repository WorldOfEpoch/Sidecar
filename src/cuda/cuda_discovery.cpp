#include "sidecar/hardware/native_components.hpp"

#include <cuda_runtime.h>

#include <array>
#include <iomanip>
#include <sstream>

namespace sidecar::hardware {
namespace {

std::string FormatCudaUuid(const cudaUUID_t& uuid) {
    static constexpr std::array<std::size_t, 5> kGroups{4, 2, 2, 2, 6};
    std::ostringstream output;
    output << "GPU-" << std::hex << std::setfill('0');
    std::size_t byte = 0;
    for (std::size_t group = 0; group < kGroups.size(); ++group) {
        if (group != 0U) output << '-';
        for (std::size_t index = 0; index < kGroups[group]; ++index, ++byte) {
            output << std::setw(2)
                   << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[byte]));
        }
    }
    return output.str();
}

std::optional<std::int64_t> ReadAttribute(int device,
                                          cudaDeviceAttr attribute,
                                          const char* name,
                                          GpuInfo& gpu) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    gpu.attributes.emplace_back(name, value);
    return value;
}

std::optional<bool> AttributeAsBool(int device,
                                    cudaDeviceAttr attribute,
                                    const char* name,
                                    GpuInfo& gpu) {
    const auto value = ReadAttribute(device, attribute, name, gpu);
    if (!value.has_value()) return std::nullopt;
    return *value != 0;
}

}  // namespace

void DiscoverCudaDevices(MachineSnapshot& snapshot, std::vector<std::string>& warnings) {
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        snapshot.cuda_runtime_version = runtime_version;
    } else {
        (void)cudaGetLastError();
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        snapshot.cuda_driver_version = driver_version;
    } else {
        (void)cudaGetLastError();
    }

    int device_count = 0;
    const cudaError_t count_result = cudaGetDeviceCount(&device_count);
    if (count_result != cudaSuccess) {
        snapshot.cuda_available = false;
        warnings.emplace_back(std::string("CUDA device discovery unavailable: ") +
                              cudaGetErrorString(count_result));
        (void)cudaGetLastError();
        return;
    }
    snapshot.cuda_available = true;

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};
        const cudaError_t result = cudaGetDeviceProperties(&properties, device);
        if (result != cudaSuccess) {
            warnings.emplace_back("CUDA device " + std::to_string(device) +
                                  " properties unavailable: " + cudaGetErrorString(result));
            (void)cudaGetLastError();
            continue;
        }

        GpuInfo gpu;
        gpu.persistent_id = FormatCudaUuid(properties.uuid);
        gpu.model = properties.name;
        gpu.cuda_device_index = device;
        gpu.vram_bytes = properties.totalGlobalMem;
        gpu.compute_capability_major = static_cast<std::uint32_t>(properties.major);
        gpu.compute_capability_minor = static_cast<std::uint32_t>(properties.minor);
        gpu.pci_domain = static_cast<std::uint32_t>(properties.pciDomainID);
        gpu.pci_bus = static_cast<std::uint32_t>(properties.pciBusID);
        gpu.pci_device = static_cast<std::uint32_t>(properties.pciDeviceID);
        gpu.async_engine_count = static_cast<std::uint32_t>(properties.asyncEngineCount);
        gpu.concurrent_kernels = properties.concurrentKernels != 0;
        gpu.unified_addressing = properties.unifiedAddressing != 0;
        gpu.can_map_host_memory = properties.canMapHostMemory != 0;
        gpu.managed_memory = properties.managedMemory != 0;
        gpu.pageable_memory_access = properties.pageableMemoryAccess != 0;
        gpu.concurrent_managed_access = properties.concurrentManagedAccess != 0;
        gpu.host_native_atomic_supported = properties.hostNativeAtomicSupported != 0;
        gpu.memory_pools_supported = AttributeAsBool(
            device, cudaDevAttrMemoryPoolsSupported, "memory_pools_supported", gpu);
        (void)ReadAttribute(device, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                            "pageable_memory_access_uses_host_page_tables", gpu);
        (void)ReadAttribute(device, cudaDevAttrDirectManagedMemAccessFromHost,
                            "direct_managed_memory_access_from_host", gpu);
        (void)ReadAttribute(device, cudaDevAttrConcurrentManagedAccess,
                            "concurrent_managed_access", gpu);
        (void)ReadAttribute(device, cudaDevAttrHostNativeAtomicSupported,
                            "host_native_atomic_supported", gpu);
        gpu.discovered_by_cuda = true;
        snapshot.gpus.push_back(std::move(gpu));
    }
}

}  // namespace sidecar::hardware

