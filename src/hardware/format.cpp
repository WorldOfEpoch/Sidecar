#include "sidecar/hardware/format.hpp"

#include "sidecar/version.hpp"

#include <algorithm>
#include <iomanip>
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

template <typename T>
void WriteOptionalInteger(std::ostringstream& output, const std::optional<T>& value) {
    if (value.has_value()) output << static_cast<std::int64_t>(*value);
    else output << "null";
}

void WriteOptionalBool(std::ostringstream& output, const std::optional<bool>& value) {
    if (value.has_value()) output << (*value ? "true" : "false");
    else output << "null";
}

void WriteStrings(std::ostringstream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << EscapeJson(values[index]);
    }
    output << ']';
}

std::string FormatBytes(const std::optional<std::uint64_t>& bytes) {
    if (!bytes.has_value()) return "unknown";
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << static_cast<double>(*bytes) / kGiB << " GiB (" << *bytes << " bytes)";
    return output.str();
}

template <typename T>
std::string NumberOrUnknown(const std::optional<T>& value) {
    return value.has_value() ? std::to_string(*value) : "unknown";
}

std::string PciAddress(const GpuInfo& gpu) {
    if (!gpu.pci_domain.has_value() || !gpu.pci_bus.has_value() ||
        !gpu.pci_device.has_value()) {
        return "unknown";
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(4) << *gpu.pci_domain << ':'
           << std::setw(2) << *gpu.pci_bus << ':' << std::setw(2) << *gpu.pci_device
           << ".0";
    return output.str();
}

void WriteGpu(std::ostringstream& output, const GpuInfo& gpu) {
    output << "{\"persistent_id\":" << EscapeJson(gpu.persistent_id)
           << ",\"model\":" << EscapeJson(gpu.model)
           << ",\"cuda_device_index\":";
    WriteOptionalInteger(output, gpu.cuda_device_index);
    output << ",\"vram_bytes\":";
    WriteOptionalInteger(output, gpu.vram_bytes);
    output << ",\"compute_capability\":{"
           << "\"major\":";
    WriteOptionalInteger(output, gpu.compute_capability_major);
    output << ",\"minor\":";
    WriteOptionalInteger(output, gpu.compute_capability_minor);
    output << "},\"pci\":{"
           << "\"domain\":";
    WriteOptionalInteger(output, gpu.pci_domain);
    output << ",\"bus\":";
    WriteOptionalInteger(output, gpu.pci_bus);
    output << ",\"device\":";
    WriteOptionalInteger(output, gpu.pci_device);
    output << ",\"current_generation\":";
    WriteOptionalInteger(output, gpu.negotiated_pcie_generation);
    output << ",\"maximum_generation\":";
    WriteOptionalInteger(output, gpu.maximum_pcie_generation);
    output << ",\"current_width\":";
    WriteOptionalInteger(output, gpu.negotiated_lane_width);
    output << ",\"maximum_width\":";
    WriteOptionalInteger(output, gpu.maximum_lane_width);
    output << "},\"capabilities\":{"
           << "\"async_engine_count\":";
    WriteOptionalInteger(output, gpu.async_engine_count);
    output << ",\"concurrent_kernels\":";
    WriteOptionalBool(output, gpu.concurrent_kernels);
    output << ",\"unified_addressing\":";
    WriteOptionalBool(output, gpu.unified_addressing);
    output << ",\"can_map_host_memory\":";
    WriteOptionalBool(output, gpu.can_map_host_memory);
    output << ",\"managed_memory\":";
    WriteOptionalBool(output, gpu.managed_memory);
    output << ",\"pageable_memory_access\":";
    WriteOptionalBool(output, gpu.pageable_memory_access);
    output << ",\"concurrent_managed_access\":";
    WriteOptionalBool(output, gpu.concurrent_managed_access);
    output << ",\"host_native_atomic_supported\":";
    WriteOptionalBool(output, gpu.host_native_atomic_supported);
    output << ",\"memory_pools_supported\":";
    WriteOptionalBool(output, gpu.memory_pools_supported);
    output << "},\"snapshot\":{"
           << "\"power_limit_watts\":";
    if (gpu.power_limit_watts.has_value()) output << *gpu.power_limit_watts;
    else output << "null";
    output << ",\"power_draw_watts\":";
    if (gpu.power_draw_watts.has_value()) output << *gpu.power_draw_watts;
    else output << "null";
    output << ",\"temperature_c\":";
    if (gpu.temperature_c.has_value()) output << *gpu.temperature_c;
    else output << "null";
    output << ",\"graphics_clock_mhz\":";
    WriteOptionalInteger(output, gpu.graphics_clock_mhz);
    output << ",\"memory_clock_mhz\":";
    WriteOptionalInteger(output, gpu.memory_clock_mhz);
    output << ",\"vram_used_bytes\":";
    WriteOptionalInteger(output, gpu.vram_used_bytes);
    output << "},\"sources\":{"
           << "\"cuda\":" << (gpu.discovered_by_cuda ? "true" : "false")
           << ",\"nvml\":" << (gpu.enriched_by_nvml ? "true" : "false")
           << "},\"location_paths\":";
    WriteStrings(output, gpu.location_paths);
    auto attributes = gpu.attributes;
    std::sort(attributes.begin(), attributes.end());
    output << ",\"attributes\":{";
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        if (index != 0U) output << ',';
        output << EscapeJson(attributes[index].first) << ':' << attributes[index].second;
    }
    output << "}}";
}

void WriteStorage(std::ostringstream& output, const StorageDeviceInfo& storage) {
    output << "{\"persistent_id\":" << EscapeJson(storage.persistent_id)
           << ",\"model\":" << EscapeJson(storage.model)
           << ",\"firmware\":" << EscapeJson(storage.firmware)
           << ",\"capacity_bytes\":";
    WriteOptionalInteger(output, storage.capacity_bytes);
    output << ",\"bus_type\":" << EscapeJson(storage.bus_type)
           << ",\"physical_disk_number\":";
    WriteOptionalInteger(output, storage.physical_disk_number);
    output << ",\"numa_node\":";
    WriteOptionalInteger(output, storage.numa_node);
    output << ",\"topology_confidence\":"
           << EscapeJson(ToString(storage.topology_confidence))
           << ",\"topology_source\":" << EscapeJson(storage.topology_source)
           << ",\"location_paths\":";
    WriteStrings(output, storage.location_paths);
    output << ",\"volumes\":[";
    for (std::size_t index = 0; index < storage.volumes.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& volume = storage.volumes[index];
        output << "{\"volume_name\":" << EscapeJson(volume.volume_name)
               << ",\"filesystem\":" << EscapeJson(volume.filesystem)
               << ",\"mount_points\":";
        WriteStrings(output, volume.mount_points);
        output << '}';
    }
    output << "]}";
}

}  // namespace

std::string DiscoveryReportToJson(const DiscoveryReport& report) {
    const auto& snapshot = report.snapshot;
    std::ostringstream output;
    output << "{\"schema\":\"sidecar.hardware.discovery.v1\",\"machine_identity\":{"
           << "\"version\":" << report.identity.version
           << ",\"quality\":" << EscapeJson(ToString(report.identity.quality))
           << ",\"machine_hash\":" << EscapeJson(report.identity.machine_hash)
           << ",\"basis\":";
    WriteStrings(output, report.identity.basis);
    output << "},\"operating_system\":{"
           << "\"name\":" << EscapeJson(snapshot.operating_system.name)
           << ",\"version\":" << EscapeJson(snapshot.operating_system.version)
           << ",\"build\":" << EscapeJson(snapshot.operating_system.build)
           << ",\"architecture\":" << EscapeJson(snapshot.operating_system.architecture)
           << ",\"host_name\":" << EscapeJson(snapshot.operating_system.host_name)
           << "},\"cpu\":{"
           << "\"vendor\":" << EscapeJson(snapshot.cpu.vendor)
           << ",\"model\":" << EscapeJson(snapshot.cpu.model)
           << ",\"architecture\":" << EscapeJson(snapshot.cpu.architecture)
           << ",\"physical_core_count\":";
    WriteOptionalInteger(output, snapshot.cpu.physical_core_count);
    output << ",\"logical_processor_count\":";
    WriteOptionalInteger(output, snapshot.cpu.logical_processor_count);
    output << ",\"processor_group_count\":";
    WriteOptionalInteger(output, snapshot.cpu.processor_group_count);
    output << ",\"numa_node_count\":";
    WriteOptionalInteger(output, snapshot.cpu.numa_node_count);
    output << ",\"package_count\":";
    WriteOptionalInteger(output, snapshot.cpu.package_count);
    output << ",\"numa_nodes\":[";
    for (std::size_t index = 0; index < snapshot.cpu.numa_nodes.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& node = snapshot.cpu.numa_nodes[index];
        output << "{\"node_id\":" << node.node_id << ",\"processor_groups\":[";
        for (std::size_t group = 0; group < node.processor_groups.size(); ++group) {
            if (group != 0U) output << ',';
            output << node.processor_groups[group];
        }
        output << "],\"available_memory_bytes\":" << node.available_memory_bytes << '}';
    }
    output << "]},\"memory\":{"
           << "\"installed_physical_bytes\":";
    WriteOptionalInteger(output, snapshot.memory.installed_physical_bytes);
    output << ",\"visible_physical_bytes\":";
    WriteOptionalInteger(output, snapshot.memory.visible_physical_bytes);
    output << ",\"available_physical_bytes\":";
    WriteOptionalInteger(output, snapshot.memory.available_physical_bytes);
    output << ",\"speed_mt_s\":";
    WriteOptionalInteger(output, snapshot.memory.speed_mt_s);
    output << ",\"channel_configuration\":";
    if (snapshot.memory.channel_configuration.has_value())
        output << EscapeJson(*snapshot.memory.channel_configuration);
    else output << "null";
    output << ",\"modules\":[";
    for (std::size_t index = 0; index < snapshot.memory.modules.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& module = snapshot.memory.modules[index];
        output << "{\"locator\":" << EscapeJson(module.locator)
               << ",\"manufacturer\":" << EscapeJson(module.manufacturer)
               << ",\"part_number\":" << EscapeJson(module.part_number)
               << ",\"capacity_bytes\":";
        WriteOptionalInteger(output, module.capacity_bytes);
        output << ",\"speed_mt_s\":";
        WriteOptionalInteger(output, module.speed_mt_s);
        output << ",\"configured_speed_mt_s\":";
        WriteOptionalInteger(output, module.configured_speed_mt_s);
        output << '}';
    }
    output << "]},\"platform\":{"
           << "\"system_manufacturer\":" << EscapeJson(snapshot.platform.system_manufacturer)
           << ",\"system_product\":" << EscapeJson(snapshot.platform.system_product)
           << ",\"board_manufacturer\":" << EscapeJson(snapshot.platform.board_manufacturer)
           << ",\"board_product\":" << EscapeJson(snapshot.platform.board_product)
           << ",\"bios_vendor\":" << EscapeJson(snapshot.platform.bios_vendor)
           << ",\"bios_version\":" << EscapeJson(snapshot.platform.bios_version)
           << ",\"smbios_version\":" << EscapeJson(snapshot.platform.smbios_version)
           << "},\"cuda\":{"
           << "\"available\":" << (snapshot.cuda_available ? "true" : "false")
           << ",\"runtime_version\":";
    WriteOptionalInteger(output, snapshot.cuda_runtime_version);
    output << ",\"driver_version\":";
    WriteOptionalInteger(output, snapshot.cuda_driver_version);
    output << "},\"nvml\":{"
           << "\"available\":" << (snapshot.nvml_available ? "true" : "false")
           << ",\"nvidia_driver_version\":" << EscapeJson(snapshot.nvidia_driver_version)
           << "},\"gpus\":[";
    auto gpu_order = snapshot.gpus;
    std::sort(gpu_order.begin(), gpu_order.end(), [](const GpuInfo& left, const GpuInfo& right) {
        return left.persistent_id < right.persistent_id;
    });
    for (std::size_t index = 0; index < gpu_order.size(); ++index) {
        if (index != 0U) output << ',';
        WriteGpu(output, gpu_order[index]);
    }
    output << "],\"storage\":{\"discovery_available\":"
           << (snapshot.storage_discovery_available ? "true" : "false")
           << "},\"storage_devices\":[";
    auto storage_order = snapshot.storage_devices;
    std::sort(storage_order.begin(), storage_order.end(),
              [](const StorageDeviceInfo& left, const StorageDeviceInfo& right) {
                  return left.persistent_id < right.persistent_id;
              });
    for (std::size_t index = 0; index < storage_order.size(); ++index) {
        if (index != 0U) output << ',';
        WriteStorage(output, storage_order[index]);
    }
    output << "],\"warnings\":";
    WriteStrings(output, report.warnings);
    output << '}';
    return output.str();
}

std::string TopologyToJson(const DiscoveryReport& report) {
    auto nodes = report.snapshot.topology;
    std::sort(nodes.begin(), nodes.end(), [](const TopologyNode& left, const TopologyNode& right) {
        return left.node_id < right.node_id;
    });
    std::ostringstream output;
    output << "{\"schema\":\"sidecar.hardware.topology.v1\",\"machine_hash\":"
           << EscapeJson(report.identity.machine_hash)
           << ",\"topology_kind\":\"DISCOVERED\",\"empirical_contention_topology_available\":false,\"nodes\":[";
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& node = nodes[index];
        output << "{\"node_id\":" << EscapeJson(node.node_id) << ",\"parent_id\":";
        if (node.parent_id.has_value()) output << EscapeJson(*node.parent_id);
        else output << "null";
        output << ",\"kind\":" << EscapeJson(node.kind)
               << ",\"label\":" << EscapeJson(node.label)
               << ",\"confidence\":" << EscapeJson(ToString(node.confidence))
               << ",\"source\":" << EscapeJson(node.source)
               << ",\"location_paths\":";
        WriteStrings(output, node.location_paths);
        auto attributes = node.attributes;
        std::sort(attributes.begin(), attributes.end());
        output << ",\"attributes\":{";
        for (std::size_t attribute = 0; attribute < attributes.size(); ++attribute) {
            if (attribute != 0U) output << ',';
            output << EscapeJson(attributes[attribute].first) << ':'
                   << EscapeJson(attributes[attribute].second);
        }
        output << "}}";
    }
    output << "]}";
    return output.str();
}

std::string FormatSystemInformation(const DiscoveryReport& report) {
    const auto& snapshot = report.snapshot;
    const auto sidecar_version = sidecar::CurrentVersionInfo();
    std::ostringstream output;
    output << "SIDECAR SYSTEM INFORMATION\n\n"
           << "Machine\n"
           << "  Machine hash: " << report.identity.machine_hash << '\n'
           << "  Identity version: " << report.identity.version << '\n'
           << "  Identity quality: " << ToString(report.identity.quality) << '\n'
           << "  Identity basis: ";
    for (std::size_t index = 0; index < report.identity.basis.size(); ++index) {
        if (index != 0U) output << ", ";
        output << report.identity.basis[index];
    }
    output << "\n\nOperating System\n"
           << "  " << snapshot.operating_system.name << '\n'
           << "  Version: " << snapshot.operating_system.version << '\n'
           << "  Build: " << snapshot.operating_system.build << '\n'
           << "  Architecture: " << snapshot.operating_system.architecture << '\n'
           << "\nPlatform\n"
           << "  System: " << snapshot.platform.system_manufacturer << ' '
           << snapshot.platform.system_product << '\n'
           << "  Motherboard: " << snapshot.platform.board_manufacturer << ' '
           << snapshot.platform.board_product << '\n'
           << "  BIOS/UEFI: " << snapshot.platform.bios_vendor << ' '
           << snapshot.platform.bios_version << '\n'
           << "  SMBIOS: " << snapshot.platform.smbios_version << '\n'
           << "\nCPU\n"
           << "  " << snapshot.cpu.model << '\n'
           << "  Vendor: " << snapshot.cpu.vendor << '\n'
           << "  Physical cores: " << NumberOrUnknown(snapshot.cpu.physical_core_count) << '\n'
           << "  Logical processors: " << NumberOrUnknown(snapshot.cpu.logical_processor_count) << '\n'
           << "  Processor groups: " << NumberOrUnknown(snapshot.cpu.processor_group_count) << '\n'
           << "  Packages: " << NumberOrUnknown(snapshot.cpu.package_count) << '\n'
           << "  NUMA nodes: " << NumberOrUnknown(snapshot.cpu.numa_node_count) << '\n'
           << "\nMemory\n"
           << "  Installed: " << FormatBytes(snapshot.memory.installed_physical_bytes) << '\n'
           << "  OS-visible: " << FormatBytes(snapshot.memory.visible_physical_bytes) << '\n'
           << "  Available: " << FormatBytes(snapshot.memory.available_physical_bytes) << '\n'
           << "  Speed: " << NumberOrUnknown(snapshot.memory.speed_mt_s) << " MT/s\n"
           << "  Channel configuration: "
           << snapshot.memory.channel_configuration.value_or("unknown") << '\n'
           << "  Reported populated modules: " << snapshot.memory.modules.size() << '\n';

    for (std::size_t index = 0; index < snapshot.gpus.size(); ++index) {
        const auto& gpu = snapshot.gpus[index];
        output << "\nGPU " << index << "\n"
               << "  " << gpu.model << '\n'
               << "  Persistent ID: " << gpu.persistent_id << '\n'
               << "  VRAM: " << FormatBytes(gpu.vram_bytes) << '\n'
               << "  Compute capability: " << NumberOrUnknown(gpu.compute_capability_major)
               << '.' << NumberOrUnknown(gpu.compute_capability_minor) << '\n'
               << "  PCI: " << PciAddress(gpu) << '\n'
               << "  PCIe current/max: Gen " << NumberOrUnknown(gpu.negotiated_pcie_generation)
               << " x" << NumberOrUnknown(gpu.negotiated_lane_width) << " / Gen "
               << NumberOrUnknown(gpu.maximum_pcie_generation) << " x"
               << NumberOrUnknown(gpu.maximum_lane_width) << '\n'
               << "  Discovery: CUDA=" << (gpu.discovered_by_cuda ? "yes" : "no")
               << ", NVML=" << (gpu.enriched_by_nvml ? "yes" : "no") << '\n';
    }

    output << "\nStorage\n";
    if (snapshot.storage_devices.empty()) output << "  No physical storage devices discovered\n";
    for (std::size_t index = 0; index < snapshot.storage_devices.size(); ++index) {
        const auto& storage = snapshot.storage_devices[index];
        output << "  Device " << index << ": " << storage.model << '\n'
               << "    Bus: " << storage.bus_type << '\n'
               << "    Capacity: " << FormatBytes(storage.capacity_bytes) << '\n'
               << "    Firmware: " << (storage.firmware.empty() ? "unknown" : storage.firmware) << '\n'
               << "    Topology confidence: " << ToString(storage.topology_confidence) << '\n';
        for (const auto& path : storage.location_paths) output << "    Location: " << path << '\n';
    }

    output << "\nSidecar\n"
           << "  Version: " << sidecar_version.version << '\n'
           << "  Spec: " << sidecar_version.spec_version << '\n'
           << "  Git commit: " << sidecar_version.git_commit << '\n'
           << "  CUDA available: " << (snapshot.cuda_available ? "yes" : "no") << '\n'
           << "  NVML available: " << (snapshot.nvml_available ? "yes" : "no") << '\n'
           << "  NVIDIA driver: "
           << (snapshot.nvidia_driver_version.empty() ? "unknown" : snapshot.nvidia_driver_version)
           << '\n';
    if (!report.warnings.empty()) {
        output << "\nWarnings\n";
        for (const auto& warning : report.warnings) output << "  - " << warning << '\n';
    }
    return output.str();
}

std::string FormatTopology(const DiscoveryReport& report) {
    auto nodes = report.snapshot.topology;
    std::sort(nodes.begin(), nodes.end(), [](const TopologyNode& left, const TopologyNode& right) {
        const auto node_rank = [](const std::string& kind) {
            if (kind == "MACHINE") return 0;
            if (kind == "NUMA_NODE") return 1;
            if (kind == "CPU_PACKAGE") return 2;
            if (kind == "GPU") return 3;
            if (kind == "STORAGE_DEVICE") return 4;
            return 5;
        };
        const int left_rank = node_rank(left.kind);
        const int right_rank = node_rank(right.kind);
        return left_rank == right_rank ? left.node_id < right.node_id
                                       : left_rank < right_rank;
    });
    std::ostringstream output;
    output << "SIDECAR DISCOVERED TOPOLOGY\n"
           << "Machine " << report.identity.machine_hash << "\n"
           << "Empirical contention topology: NOT YET MEASURED\n\n";
    for (const auto& node : nodes) {
        output << (node.parent_id.has_value() ? "+-- " : "") << node.kind << ": " << node.label
               << " [" << ToString(node.confidence) << ", " << node.source << "]\n";
        if (node.parent_id.has_value()) output << "    Parent: " << *node.parent_id << '\n';
        for (const auto& path : node.location_paths) output << "    Location: " << path << '\n';
    }
    if (nodes.empty()) output << "No topology nodes were discoverable.\n";
    output << "\nTopology relationships are reported only to the confidence supported by Windows APIs.\n";
    return output.str();
}

}  // namespace sidecar::hardware
