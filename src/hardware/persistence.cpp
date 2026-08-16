#include "sidecar/hardware/persistence.hpp"

#include "sidecar/hardware/format.hpp"

#include <algorithm>
#include <sstream>

namespace sidecar::hardware {
namespace {

std::string EscapeJson(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    constexpr char kHex[] = "0123456789abcdef";
                    output << "\\u00" << kHex[character >> 4U] << kHex[character & 0x0FU];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
    return output.str();
}

std::string StringsToJson(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << EscapeJson(values[index]);
    }
    output << ']';
    return output.str();
}

std::string AttributesToJson(
    const std::vector<std::pair<std::string, std::int64_t>>& attributes) {
    auto sorted = attributes;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream output;
    output << '{';
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (index != 0U) output << ',';
        output << EscapeJson(sorted[index].first) << ':' << sorted[index].second;
    }
    output << '}';
    return output.str();
}

std::string VolumesToJson(const std::vector<VolumeInfo>& volumes) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < volumes.size(); ++index) {
        if (index != 0U) output << ',';
        output << "{\"volume_name\":" << EscapeJson(volumes[index].volume_name)
               << ",\"filesystem\":" << EscapeJson(volumes[index].filesystem)
               << ",\"mount_points\":" << StringsToJson(volumes[index].mount_points) << '}';
    }
    output << ']';
    return output.str();
}

template <typename T>
std::optional<std::int64_t> ToDatabaseInteger(const std::optional<T>& value) {
    if (!value.has_value()) return std::nullopt;
    return static_cast<std::int64_t>(*value);
}

std::optional<std::int64_t> ToDatabaseBoolean(const std::optional<bool>& value) {
    if (!value.has_value()) return std::nullopt;
    return *value ? 1 : 0;
}

}  // namespace

void PersistDiscovery(database::Database& database, const DiscoveryReport& report) {
    const auto& snapshot = report.snapshot;
    database::HardwareProfileInput profile;
    profile.machine_hash = report.identity.machine_hash;
    profile.host_name = snapshot.operating_system.host_name;
    profile.os_name = snapshot.operating_system.name;
    profile.os_version = snapshot.operating_system.version;
    profile.os_build = snapshot.operating_system.build;
    profile.cpu_model = snapshot.cpu.model;
    profile.physical_core_count = ToDatabaseInteger(snapshot.cpu.physical_core_count);
    profile.logical_core_count = ToDatabaseInteger(snapshot.cpu.logical_processor_count);
    profile.numa_node_count = ToDatabaseInteger(snapshot.cpu.numa_node_count);
    profile.installed_ram_bytes = ToDatabaseInteger(snapshot.memory.installed_physical_bytes);
    profile.available_ram_bytes = ToDatabaseInteger(snapshot.memory.available_physical_bytes);
    profile.discovery_json = DiscoveryReportToJson(report);
    profile.identity_version = report.identity.version;
    profile.identity_quality = ToString(report.identity.quality);
    profile.identity_basis_json = StringsToJson(report.identity.basis);
    profile.system_manufacturer = snapshot.platform.system_manufacturer;
    profile.system_product = snapshot.platform.system_product;
    profile.smbios_version = snapshot.platform.smbios_version;
    profile.cpu_vendor = snapshot.cpu.vendor;
    profile.cpu_architecture = snapshot.cpu.architecture;
    profile.processor_group_count = ToDatabaseInteger(snapshot.cpu.processor_group_count);
    profile.cpu_package_count = ToDatabaseInteger(snapshot.cpu.package_count);
    profile.ram_speed_mt_s = ToDatabaseInteger(snapshot.memory.speed_mt_s);
    profile.motherboard_manufacturer = snapshot.platform.board_manufacturer;
    profile.motherboard_model = snapshot.platform.board_product;
    profile.firmware_revision = snapshot.platform.bios_version;
    profile.topology_json = TopologyToJson(report);
    profile.snapshot_json = profile.discovery_json;

    std::vector<database::GpuDeviceInput> gpus;
    gpus.reserve(snapshot.gpus.size());
    for (const auto& gpu : snapshot.gpus) {
        database::GpuDeviceInput input;
        input.persistent_id = gpu.persistent_id;
        input.model = gpu.model;
        input.cuda_device_index = ToDatabaseInteger(gpu.cuda_device_index);
        input.vram_bytes = ToDatabaseInteger(gpu.vram_bytes);
        input.pci_domain = ToDatabaseInteger(gpu.pci_domain);
        input.pci_bus = ToDatabaseInteger(gpu.pci_bus);
        input.pci_device = ToDatabaseInteger(gpu.pci_device);
        input.negotiated_pcie_generation = ToDatabaseInteger(gpu.negotiated_pcie_generation);
        input.maximum_pcie_generation = ToDatabaseInteger(gpu.maximum_pcie_generation);
        input.negotiated_lane_width = ToDatabaseInteger(gpu.negotiated_lane_width);
        input.maximum_lane_width = ToDatabaseInteger(gpu.maximum_lane_width);
        input.compute_capability_major = ToDatabaseInteger(gpu.compute_capability_major);
        input.compute_capability_minor = ToDatabaseInteger(gpu.compute_capability_minor);
        input.async_engine_count = ToDatabaseInteger(gpu.async_engine_count);
        input.concurrent_kernels = ToDatabaseBoolean(gpu.concurrent_kernels);
        input.unified_addressing = ToDatabaseBoolean(gpu.unified_addressing);
        input.can_map_host_memory = ToDatabaseBoolean(gpu.can_map_host_memory);
        input.managed_memory = ToDatabaseBoolean(gpu.managed_memory);
        input.pageable_memory_access = ToDatabaseBoolean(gpu.pageable_memory_access);
        auto attributes = gpu.attributes;
        if (gpu.vram_used_bytes) attributes.emplace_back("nvml.vram_used_bytes",
            static_cast<std::int64_t>(*gpu.vram_used_bytes));
        if (gpu.graphics_clock_mhz) attributes.emplace_back("nvml.graphics_clock_mhz",
            static_cast<std::int64_t>(*gpu.graphics_clock_mhz));
        if (gpu.memory_clock_mhz) attributes.emplace_back("nvml.memory_clock_mhz",
            static_cast<std::int64_t>(*gpu.memory_clock_mhz));
        if (gpu.power_draw_watts) attributes.emplace_back("nvml.power_draw_milliwatts",
            static_cast<std::int64_t>(*gpu.power_draw_watts * 1000.0));
        input.attributes_json = AttributesToJson(attributes);
        input.location_paths_json = StringsToJson(gpu.location_paths);
        gpus.push_back(std::move(input));
    }

    std::vector<database::StorageDeviceInput> storage_devices;
    storage_devices.reserve(snapshot.storage_devices.size());
    for (const auto& storage : snapshot.storage_devices) {
        database::StorageDeviceInput input;
        input.persistent_id = storage.persistent_id;
        input.model = storage.model;
        input.firmware = storage.firmware;
        input.capacity_bytes = ToDatabaseInteger(storage.capacity_bytes);
        std::vector<std::string> filesystems;
        for (const auto& volume : storage.volumes) {
            if (!volume.filesystem.empty() &&
                std::find(filesystems.begin(), filesystems.end(), volume.filesystem) ==
                    filesystems.end()) {
                filesystems.push_back(volume.filesystem);
            }
        }
        for (std::size_t index = 0; index < filesystems.size(); ++index) {
            if (index != 0U) input.filesystem += ',';
            input.filesystem += filesystems[index];
        }
        input.bus_type = storage.bus_type;
        input.pci_path = storage.location_paths.empty() ? "" : storage.location_paths.front();
        input.numa_node = ToDatabaseInteger(storage.numa_node);
        input.topology_json = StringsToJson(storage.location_paths);
        input.volumes_json = VolumesToJson(storage.volumes);
        input.topology_confidence = ToString(storage.topology_confidence);
        input.topology_source = storage.topology_source;
        storage_devices.push_back(std::move(input));
    }

    database.RefreshHardwareInventory(profile, gpus, storage_devices,
                                      snapshot.cuda_available || snapshot.nvml_available,
                                      snapshot.storage_discovery_available);
}

}  // namespace sidecar::hardware
