#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sidecar::hardware {

inline constexpr std::uint32_t kMachineIdentityVersion = 1;

enum class IdentityQuality {
    Strong,
    Moderate,
    Fallback,
};

enum class DiscoveryConfidence {
    DirectlyReported,
    Derived,
    Unknown,
};

[[nodiscard]] const char* ToString(IdentityQuality quality) noexcept;
[[nodiscard]] const char* ToString(DiscoveryConfidence confidence) noexcept;

struct MachineIdentityInput {
    std::optional<std::string> system_uuid;
    std::optional<std::string> system_manufacturer;
    std::optional<std::string> system_product;
    std::optional<std::string> system_serial;
    std::optional<std::string> board_manufacturer;
    std::optional<std::string> board_product;
    std::optional<std::string> board_serial;
    // Windows MachineGuid is an installation identity, not physical hardware.
    // It is considered only when usable SMBIOS identity is unavailable.
    std::optional<std::string> fallback_installation_id;
};

struct MachineIdentity {
    std::uint32_t version{kMachineIdentityVersion};
    IdentityQuality quality{IdentityQuality::Fallback};
    std::string machine_hash;
    std::string canonical_material;
    std::vector<std::string> basis;
    std::vector<std::string> warnings;
};

struct OperatingSystemInfo {
    std::string name;
    std::string version;
    std::string build;
    std::string architecture;
    std::string host_name;
};

struct NumaNodeInfo {
    std::uint32_t node_id{0};
    std::vector<std::uint16_t> processor_groups;
    std::uint64_t available_memory_bytes{0};
};

struct CpuInfo {
    std::string vendor;
    std::string model;
    std::string architecture;
    std::optional<std::uint32_t> physical_core_count;
    std::optional<std::uint32_t> logical_processor_count;
    std::optional<std::uint32_t> processor_group_count;
    std::optional<std::uint32_t> numa_node_count;
    std::optional<std::uint32_t> package_count;
    std::vector<NumaNodeInfo> numa_nodes;
};

struct MemoryModuleInfo {
    std::string locator;
    std::string manufacturer;
    std::string part_number;
    std::optional<std::uint64_t> capacity_bytes;
    std::optional<std::uint32_t> speed_mt_s;
    std::optional<std::uint32_t> configured_speed_mt_s;
};

struct MemoryInfo {
    std::optional<std::uint64_t> installed_physical_bytes;
    std::optional<std::uint64_t> visible_physical_bytes;
    std::optional<std::uint64_t> available_physical_bytes;
    std::optional<std::uint32_t> speed_mt_s;
    std::optional<std::string> channel_configuration;
    std::vector<MemoryModuleInfo> modules;
};

struct PlatformInfo {
    std::string system_manufacturer;
    std::string system_product;
    std::string board_manufacturer;
    std::string board_product;
    std::string bios_vendor;
    std::string bios_version;
    std::string smbios_version;
};

struct GpuInfo {
    std::string persistent_id;
    std::string model;
    std::optional<std::int32_t> cuda_device_index;
    std::optional<std::uint64_t> vram_bytes;
    std::optional<std::uint32_t> compute_capability_major;
    std::optional<std::uint32_t> compute_capability_minor;
    std::optional<std::uint32_t> pci_domain;
    std::optional<std::uint32_t> pci_bus;
    std::optional<std::uint32_t> pci_device;
    std::optional<std::uint32_t> async_engine_count;
    std::optional<bool> concurrent_kernels;
    std::optional<bool> unified_addressing;
    std::optional<bool> can_map_host_memory;
    std::optional<bool> managed_memory;
    std::optional<bool> pageable_memory_access;
    std::optional<bool> concurrent_managed_access;
    std::optional<bool> host_native_atomic_supported;
    std::optional<bool> memory_pools_supported;
    std::optional<std::uint32_t> negotiated_pcie_generation;
    std::optional<std::uint32_t> maximum_pcie_generation;
    std::optional<std::uint32_t> negotiated_lane_width;
    std::optional<std::uint32_t> maximum_lane_width;
    std::optional<double> power_limit_watts;
    std::optional<double> power_draw_watts;
    std::optional<double> temperature_c;
    std::optional<std::uint32_t> graphics_clock_mhz;
    std::optional<std::uint32_t> memory_clock_mhz;
    std::optional<std::uint64_t> vram_used_bytes;
    std::vector<std::string> location_paths;
    std::vector<std::pair<std::string, std::int64_t>> attributes;
    bool discovered_by_cuda{false};
    bool enriched_by_nvml{false};
};

struct VolumeInfo {
    std::string volume_name;
    std::vector<std::string> mount_points;
    std::string filesystem;
};

struct StorageDeviceInfo {
    std::string persistent_id;
    std::string model;
    std::string firmware;
    std::optional<std::uint64_t> capacity_bytes;
    std::string bus_type;
    std::optional<std::uint32_t> physical_disk_number;
    std::vector<std::string> location_paths;
    std::optional<std::uint32_t> numa_node;
    DiscoveryConfidence topology_confidence{DiscoveryConfidence::Unknown};
    std::string topology_source;
    std::vector<VolumeInfo> volumes;
};

struct TopologyNode {
    std::string node_id;
    std::optional<std::string> parent_id;
    std::string kind;
    std::string label;
    DiscoveryConfidence confidence{DiscoveryConfidence::Unknown};
    std::string source;
    std::vector<std::string> location_paths;
    std::vector<std::pair<std::string, std::string>> attributes;
};

struct MachineSnapshot {
    OperatingSystemInfo operating_system;
    CpuInfo cpu;
    MemoryInfo memory;
    PlatformInfo platform;
    std::vector<GpuInfo> gpus;
    std::vector<StorageDeviceInfo> storage_devices;
    bool storage_discovery_available{false};
    std::vector<TopologyNode> topology;
    bool cuda_available{false};
    std::optional<std::int32_t> cuda_runtime_version;
    std::optional<std::int32_t> cuda_driver_version;
    bool nvml_available{false};
    std::string nvidia_driver_version;
};

struct RawDiscovery {
    MachineIdentityInput identity_input;
    MachineSnapshot snapshot;
    std::vector<std::string> warnings;
};

struct DiscoveryReport {
    MachineIdentity identity;
    MachineSnapshot snapshot;
    std::vector<std::string> warnings;
};

class IHardwareDiscoveryProvider {
public:
    virtual ~IHardwareDiscoveryProvider() = default;
    [[nodiscard]] virtual RawDiscovery Discover() = 0;
};

class HardwareDiscoveryService final {
public:
    explicit HardwareDiscoveryService(IHardwareDiscoveryProvider& provider) noexcept;
    [[nodiscard]] DiscoveryReport Discover() const;

private:
    IHardwareDiscoveryProvider& provider_;
};

[[nodiscard]] std::unique_ptr<IHardwareDiscoveryProvider> CreateNativeDiscoveryProvider();

}  // namespace sidecar::hardware
