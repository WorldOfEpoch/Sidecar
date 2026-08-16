#include "sidecar/hardware/hardware.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/hardware/native_components.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <initguid.h>
#include <cfgmgr32.h>
#include <devpkey.h>
#include <devguid.h>
#include <setupapi.h>
#include <winioctl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sidecar::hardware {
namespace {

constexpr GUID kDiskInterfaceGuid{
    0x53f56307, 0xb6bf, 0x11d0, {0x94, 0xf2, 0x00, 0xa0, 0xc9, 0x1e, 0xfb, 0x8b}};

class Handle final {
public:
    explicit Handle(HANDLE value = INVALID_HANDLE_VALUE) noexcept : value_(value) {}
    ~Handle() { if (valid()) CloseHandle(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    [[nodiscard]] bool valid() const noexcept {
        return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
    }
    [[nodiscard]] HANDLE get() const noexcept { return value_; }
private:
    HANDLE value_;
};

class DeviceInfoSet final {
public:
    explicit DeviceInfoSet(HDEVINFO value) noexcept : value_(value) {}
    ~DeviceInfoSet() {
        if (value_ != INVALID_HANDLE_VALUE) SetupDiDestroyDeviceInfoList(value_);
    }
    DeviceInfoSet(const DeviceInfoSet&) = delete;
    DeviceInfoSet& operator=(const DeviceInfoSet&) = delete;
    [[nodiscard]] HDEVINFO get() const noexcept { return value_; }
    [[nodiscard]] bool valid() const noexcept { return value_ != INVALID_HANDLE_VALUE; }
private:
    HDEVINFO value_;
};

std::string Utf8(std::wstring_view input) {
    if (input.empty()) return {};
    const int required = WideCharToMultiByte(CP_UTF8, 0, input.data(),
                                             static_cast<int>(input.size()),
                                             nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string output(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, input.data(), static_cast<int>(input.size()),
                        output.data(), required, nullptr, nullptr);
    return output;
}

std::string Trim(std::string value) {
    const auto not_space = [](unsigned char character) { return std::isspace(character) == 0; };
    const auto first = std::find_if(value.begin(), value.end(), not_space);
    const auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::optional<std::string> ReadRegistryString(HKEY root,
                                               const wchar_t* key_path,
                                               const wchar_t* value_name) {
    DWORD type = 0;
    DWORD bytes = 0;
    if (RegGetValueW(root, key_path, value_name, RRF_RT_REG_SZ, &type, nullptr, &bytes) !=
        ERROR_SUCCESS || bytes < sizeof(wchar_t)) {
        return std::nullopt;
    }
    std::vector<wchar_t> buffer(bytes / sizeof(wchar_t) + 1U, L'\0');
    if (RegGetValueW(root, key_path, value_name, RRF_RT_REG_SZ, &type,
                     buffer.data(), &bytes) != ERROR_SUCCESS) {
        return std::nullopt;
    }
    return Trim(Utf8(buffer.data()));
}

std::string ArchitectureName(WORD architecture) {
    switch (architecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: return "x86_64";
        case PROCESSOR_ARCHITECTURE_ARM64: return "arm64";
        case PROCESSOR_ARCHITECTURE_INTEL: return "x86";
        default: return "unknown";
    }
}

std::optional<std::string> QueryDeviceInstanceId(HDEVINFO devices,
                                                  SP_DEVINFO_DATA& device) {
    DWORD required = 0;
    SetupDiGetDeviceInstanceIdW(devices, &device, nullptr, 0, &required);
    if (required == 0) return std::nullopt;
    std::vector<wchar_t> buffer(required + 1U, L'\0');
    if (!SetupDiGetDeviceInstanceIdW(devices, &device, buffer.data(),
                                     static_cast<DWORD>(buffer.size()), nullptr)) {
        return std::nullopt;
    }
    return Utf8(buffer.data());
}

std::vector<std::string> QueryLocationPaths(HDEVINFO devices,
                                             SP_DEVINFO_DATA& device) {
    DEVPROPTYPE type = 0;
    DWORD required = 0;
    SetupDiGetDevicePropertyW(devices, &device, &DEVPKEY_Device_LocationPaths,
                              &type, nullptr, 0, &required, 0);
    if (required == 0) return {};
    std::vector<BYTE> buffer(required + sizeof(wchar_t) * 2U, 0);
    if (!SetupDiGetDevicePropertyW(devices, &device, &DEVPKEY_Device_LocationPaths,
                                   &type, buffer.data(), required, nullptr, 0)) {
        return {};
    }
    std::vector<std::string> paths;
    const auto* current = reinterpret_cast<const wchar_t*>(buffer.data());
    while (*current != L'\0') {
        std::wstring_view value(current);
        paths.push_back(Utf8(value));
        current += value.size() + 1U;
    }
    return paths;
}

std::vector<std::string> QueryLocationPaths(DEVINST device_instance) {
    DEVPROPTYPE type = 0;
    ULONG bytes = 0;
    const CONFIGRET sizing = CM_Get_DevNode_PropertyW(
        device_instance, &DEVPKEY_Device_LocationPaths, &type, nullptr, &bytes, 0);
    if (sizing != CR_BUFFER_SMALL || bytes == 0) return {};
    std::vector<BYTE> buffer(bytes + sizeof(wchar_t) * 2U, 0);
    if (CM_Get_DevNode_PropertyW(device_instance, &DEVPKEY_Device_LocationPaths,
                                &type, buffer.data(), &bytes, 0) != CR_SUCCESS) {
        return {};
    }
    std::vector<std::string> paths;
    const auto* current = reinterpret_cast<const wchar_t*>(buffer.data());
    while (*current != L'\0') {
        std::wstring_view value(current);
        paths.push_back(Utf8(value));
        current += value.size() + 1U;
    }
    return paths;
}

std::vector<std::string> QueryNearestLocationPaths(DEVINST device_instance,
                                                    bool& used_ancestor) {
    DEVINST current = device_instance;
    for (unsigned depth = 0; depth < 12U; ++depth) {
        auto paths = QueryLocationPaths(current);
        if (!paths.empty()) {
            used_ancestor = depth != 0U;
            return paths;
        }
        DEVINST parent = 0;
        if (CM_Get_Parent(&parent, current, 0) != CR_SUCCESS) break;
        current = parent;
    }
    return {};
}

std::optional<DWORD> QueryDeviceDword(HDEVINFO devices,
                                      SP_DEVINFO_DATA& device,
                                      const DEVPROPKEY& key) {
    DEVPROPTYPE type = 0;
    DWORD value = 0;
    DWORD required = 0;
    if (!SetupDiGetDevicePropertyW(devices, &device, &key, &type,
                                   reinterpret_cast<PBYTE>(&value), sizeof(value),
                                   &required, 0) || type != DEVPROP_TYPE_UINT32) {
        return std::nullopt;
    }
    return value;
}

#pragma pack(push, 1)
struct RawSmbiosData {
    BYTE used_calling_method;
    BYTE major_version;
    BYTE minor_version;
    BYTE dmi_revision;
    DWORD length;
    BYTE table[1];
};
#pragma pack(pop)

std::uint16_t ReadU16(const BYTE* data) {
    return static_cast<std::uint16_t>(data[0]) |
           (static_cast<std::uint16_t>(data[1]) << 8U);
}

std::uint32_t ReadU32(const BYTE* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8U) |
           (static_cast<std::uint32_t>(data[2]) << 16U) |
           (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::string SmbiosString(const BYTE* structure,
                         std::size_t structure_length,
                         std::size_t available,
                         BYTE string_index) {
    if (string_index == 0 || structure_length >= available) return {};
    const char* current = reinterpret_cast<const char*>(structure + structure_length);
    const char* end = reinterpret_cast<const char*>(structure + available);
    BYTE index = 1;
    while (current < end && *current != '\0') {
        const char* terminator = std::find(current, end, '\0');
        if (index == string_index) return Trim(std::string(current, terminator));
        if (terminator == end) break;
        current = terminator + 1;
        ++index;
    }
    return {};
}

std::string FormatSmbiosUuid(const BYTE* uuid, bool modern_byte_order) {
    std::array<unsigned, 16> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) bytes[index] = uuid[index];
    if (modern_byte_order) {
        std::reverse(bytes.begin(), bytes.begin() + 4);
        std::reverse(bytes.begin() + 4, bytes.begin() + 6);
        std::reverse(bytes.begin() + 6, bytes.begin() + 8);
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) output << '-';
        output << std::setw(2) << bytes[index];
    }
    return output.str();
}

void DiscoverSmbios(RawDiscovery& discovery) {
    const UINT bytes = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
    if (bytes < offsetof(RawSmbiosData, table)) {
        discovery.warnings.emplace_back("Windows did not provide an SMBIOS table");
        return;
    }
    std::vector<BYTE> buffer(bytes);
    if (GetSystemFirmwareTable('RSMB', 0, buffer.data(), bytes) != bytes) {
        discovery.warnings.emplace_back("Windows SMBIOS table retrieval failed");
        return;
    }
    const auto* raw = reinterpret_cast<const RawSmbiosData*>(buffer.data());
    const std::size_t header_bytes = offsetof(RawSmbiosData, table);
    const std::size_t table_bytes = std::min<std::size_t>(raw->length, bytes - header_bytes);
    discovery.snapshot.platform.smbios_version =
        std::to_string(raw->major_version) + "." + std::to_string(raw->minor_version);

    std::size_t offset = 0;
    while (offset + 4U <= table_bytes) {
        const BYTE* structure = raw->table + offset;
        const BYTE type = structure[0];
        const BYTE length = structure[1];
        if (length < 4U || offset + length > table_bytes) break;

        std::size_t structure_end = offset + length;
        while (structure_end + 1U < table_bytes &&
               !(raw->table[structure_end] == 0 && raw->table[structure_end + 1U] == 0)) {
            ++structure_end;
        }
        const std::size_t available =
            std::min(table_bytes - offset, structure_end + 2U - offset);

        if (type == 0 && length > 5U) {
            discovery.snapshot.platform.bios_vendor =
                SmbiosString(structure, length, available, structure[4]);
            discovery.snapshot.platform.bios_version =
                SmbiosString(structure, length, available, structure[5]);
        } else if (type == 1 && length >= 8U) {
            const std::string manufacturer =
                SmbiosString(structure, length, available, structure[4]);
            const std::string product =
                SmbiosString(structure, length, available, structure[5]);
            const std::string serial =
                SmbiosString(structure, length, available, structure[7]);
            discovery.snapshot.platform.system_manufacturer = manufacturer;
            discovery.snapshot.platform.system_product = product;
            discovery.identity_input.system_manufacturer = manufacturer;
            discovery.identity_input.system_product = product;
            discovery.identity_input.system_serial = serial;
            if (length >= 0x19U) {
                const bool modern_order = raw->major_version > 2U ||
                                          (raw->major_version == 2U && raw->minor_version >= 6U);
                discovery.identity_input.system_uuid =
                    FormatSmbiosUuid(structure + 8U, modern_order);
            }
        } else if (type == 2 && length >= 8U) {
            const std::string manufacturer =
                SmbiosString(structure, length, available, structure[4]);
            const std::string product =
                SmbiosString(structure, length, available, structure[5]);
            const std::string serial =
                SmbiosString(structure, length, available, structure[7]);
            discovery.snapshot.platform.board_manufacturer = manufacturer;
            discovery.snapshot.platform.board_product = product;
            discovery.identity_input.board_manufacturer = manufacturer;
            discovery.identity_input.board_product = product;
            discovery.identity_input.board_serial = serial;
        } else if (type == 17 && length >= 0x1BU) {
            MemoryModuleInfo module;
            module.locator = SmbiosString(structure, length, available, structure[0x10]);
            module.manufacturer = SmbiosString(structure, length, available, structure[0x17]);
            module.part_number = SmbiosString(structure, length, available, structure[0x1A]);
            const std::uint16_t size = ReadU16(structure + 0x0C);
            if (size != 0U && size != 0xFFFFU) {
                if (size == 0x7FFFU && length >= 0x20U) {
                    module.capacity_bytes =
                        static_cast<std::uint64_t>(ReadU32(structure + 0x1C) & 0x7FFFFFFFU) *
                        1024ULL * 1024ULL;
                } else if ((size & 0x8000U) != 0U) {
                    module.capacity_bytes = static_cast<std::uint64_t>(size & 0x7FFFU) * 1024ULL;
                } else {
                    module.capacity_bytes = static_cast<std::uint64_t>(size) * 1024ULL * 1024ULL;
                }
            }
            const std::uint16_t speed = ReadU16(structure + 0x15);
            if (speed != 0U && speed != 0xFFFFU) module.speed_mt_s = speed;
            if (length >= 0x22U) {
                const std::uint16_t configured = ReadU16(structure + 0x20);
                if (configured != 0U && configured != 0xFFFFU) {
                    module.configured_speed_mt_s = configured;
                }
            }
            if (module.capacity_bytes.value_or(0) > 0) {
                discovery.snapshot.memory.modules.push_back(std::move(module));
            }
        }

        if (structure_end + 1U >= table_bytes) break;
        offset = structure_end + 2U;
        if (type == 127) break;
    }

    std::optional<std::uint32_t> common_speed;
    bool speeds_agree = true;
    for (const auto& module : discovery.snapshot.memory.modules) {
        const auto speed = module.configured_speed_mt_s.has_value()
                               ? module.configured_speed_mt_s
                               : module.speed_mt_s;
        if (!speed.has_value()) continue;
        if (!common_speed.has_value()) common_speed = speed;
        else if (*common_speed != *speed) speeds_agree = false;
    }
    if (speeds_agree) discovery.snapshot.memory.speed_mt_s = common_speed;
}

void DiscoverOperatingSystem(MachineSnapshot& snapshot) {
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto rtl_get_version = ntdll == nullptr
        ? nullptr
        : reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
    OSVERSIONINFOW version{};
    version.dwOSVersionInfoSize = sizeof(version);
    if (rtl_get_version != nullptr && rtl_get_version(&version) == 0) {
        snapshot.operating_system.version =
            std::to_string(version.dwMajorVersion) + "." +
            std::to_string(version.dwMinorVersion) + "." +
            std::to_string(version.dwBuildNumber);
        snapshot.operating_system.build = std::to_string(version.dwBuildNumber);
    }
    snapshot.operating_system.name =
        ReadRegistryString(HKEY_LOCAL_MACHINE,
                           L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
                           L"ProductName").value_or("Windows");
    if (version.dwBuildNumber >= 22000U &&
        snapshot.operating_system.name.rfind("Windows 10", 0) == 0U) {
        snapshot.operating_system.name.replace(0, std::string("Windows 10").size(),
                                               "Windows 11");
    }
    SYSTEM_INFO system_info{};
    GetNativeSystemInfo(&system_info);
    snapshot.operating_system.architecture =
        ArchitectureName(system_info.wProcessorArchitecture);
    std::array<wchar_t, 256> host{};
    DWORD host_size = static_cast<DWORD>(host.size());
    if (GetComputerNameExW(ComputerNameDnsHostname, host.data(), &host_size)) {
        snapshot.operating_system.host_name = Utf8(std::wstring_view(host.data(), host_size));
    }
}

void DiscoverCpu(MachineSnapshot& snapshot, std::vector<std::string>& warnings) {
    snapshot.cpu.model = ReadRegistryString(
        HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        L"ProcessorNameString").value_or("unknown");
    snapshot.cpu.vendor = ReadRegistryString(
        HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        L"VendorIdentifier").value_or("unknown");
    SYSTEM_INFO system_info{};
    GetNativeSystemInfo(&system_info);
    snapshot.cpu.architecture = ArchitectureName(system_info.wProcessorArchitecture);
    snapshot.cpu.processor_group_count = GetActiveProcessorGroupCount();
    const DWORD logical = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (logical != 0U) snapshot.cpu.logical_processor_count = logical;

    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
    if (bytes == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        warnings.emplace_back("Windows logical processor topology was unavailable");
        return;
    }
    std::vector<BYTE> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(
            RelationAll,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &bytes)) {
        warnings.emplace_back("Windows logical processor topology query failed");
        return;
    }
    std::uint32_t cores = 0;
    std::uint32_t packages = 0;
    std::map<std::uint32_t, NumaNodeInfo> nodes;
    std::size_t offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bytes) {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffer.data() + offset);
        if (info->Size == 0 || offset + info->Size > bytes) break;
        if (info->Relationship == RelationProcessorCore) {
            ++cores;
        } else if (info->Relationship == RelationProcessorPackage) {
            ++packages;
        } else if (info->Relationship == RelationNumaNode ||
                   info->Relationship == RelationNumaNodeEx) {
            auto& node = nodes[info->NumaNode.NodeNumber];
            node.node_id = info->NumaNode.NodeNumber;
            const WORD group_count = std::max<WORD>(1, info->NumaNode.GroupCount);
            for (WORD group = 0; group < group_count; ++group) {
                node.processor_groups.push_back(info->NumaNode.GroupMasks[group].Group);
            }
            ULONGLONG available = 0;
            if (GetNumaAvailableMemoryNodeEx(static_cast<USHORT>(node.node_id), &available)) {
                node.available_memory_bytes = available;
            }
        }
        offset += info->Size;
    }
    if (cores > 0U) snapshot.cpu.physical_core_count = cores;
    if (packages > 0U) snapshot.cpu.package_count = packages;
    for (auto& [id, node] : nodes) {
        (void)id;
        std::sort(node.processor_groups.begin(), node.processor_groups.end());
        node.processor_groups.erase(
            std::unique(node.processor_groups.begin(), node.processor_groups.end()),
            node.processor_groups.end());
        snapshot.cpu.numa_nodes.push_back(std::move(node));
    }
    snapshot.cpu.numa_node_count =
        static_cast<std::uint32_t>(snapshot.cpu.numa_nodes.size());
}

void DiscoverMemory(MachineSnapshot& snapshot) {
    ULONGLONG installed_kib = 0;
    if (GetPhysicallyInstalledSystemMemory(&installed_kib)) {
        snapshot.memory.installed_physical_bytes = installed_kib * 1024ULL;
    }
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        snapshot.memory.visible_physical_bytes = status.ullTotalPhys;
        snapshot.memory.available_physical_bytes = status.ullAvailPhys;
    }
    // SMBIOS does not report active memory channels reliably. DIMM count is not
    // used as a substitute, so channel_configuration intentionally remains null.
}

std::string DescriptorString(const STORAGE_DEVICE_DESCRIPTOR* descriptor,
                             DWORD offset,
                             std::size_t bytes) {
    if (offset == 0 || offset >= bytes) return {};
    const char* begin = reinterpret_cast<const char*>(descriptor) + offset;
    const char* end = reinterpret_cast<const char*>(descriptor) + bytes;
    const char* terminator = std::find(begin, end, '\0');
    return Trim(std::string(begin, terminator));
}

std::string BusTypeName(STORAGE_BUS_TYPE bus) {
    switch (bus) {
        case BusTypeNvme: return "NVMe";
        case BusTypeSata: return "SATA";
        case BusTypeAta: return "ATA";
        case BusTypeScsi: return "SCSI";
        case BusTypeUsb: return "USB";
        case BusTypeRAID: return "RAID";
        case BusTypeSas: return "SAS";
        case BusTypeVirtual: return "Virtual";
        case BusTypeFileBackedVirtual: return "FileBackedVirtual";
        default: return "Unknown";
    }
}

std::map<DWORD, std::vector<VolumeInfo>> DiscoverVolumes() {
    std::map<DWORD, std::vector<VolumeInfo>> by_disk;
    std::vector<wchar_t> volume_name(MAX_PATH + 1U, L'\0');
    HANDLE search = FindFirstVolumeW(volume_name.data(), static_cast<DWORD>(volume_name.size()));
    if (search == INVALID_HANDLE_VALUE) return by_disk;
    do {
        const std::wstring volume(volume_name.data());
        std::wstring open_path = volume;
        if (!open_path.empty() && open_path.back() == L'\\') open_path.pop_back();
        Handle handle(CreateFileW(open_path.c_str(), 0,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, 0, nullptr));
        if (!handle.valid()) continue;
        std::vector<BYTE> extent_buffer(sizeof(VOLUME_DISK_EXTENTS) +
                                        sizeof(DISK_EXTENT) * 31U);
        DWORD returned = 0;
        if (!DeviceIoControl(handle.get(), IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
                             nullptr, 0, extent_buffer.data(),
                             static_cast<DWORD>(extent_buffer.size()), &returned, nullptr)) {
            continue;
        }
        const auto* extents = reinterpret_cast<const VOLUME_DISK_EXTENTS*>(extent_buffer.data());
        VolumeInfo info;
        info.volume_name = Utf8(volume);
        DWORD mount_bytes = 0;
        GetVolumePathNamesForVolumeNameW(volume.c_str(), nullptr, 0, &mount_bytes);
        if (mount_bytes > 0) {
            std::vector<wchar_t> mounts(mount_bytes + 1U, L'\0');
            if (GetVolumePathNamesForVolumeNameW(volume.c_str(), mounts.data(),
                                                 static_cast<DWORD>(mounts.size()),
                                                 &mount_bytes)) {
                const wchar_t* current = mounts.data();
                while (*current != L'\0') {
                    std::wstring_view path(current);
                    info.mount_points.push_back(Utf8(path));
                    current += path.size() + 1U;
                }
            }
        }
        std::array<wchar_t, 64> filesystem{};
        if (GetVolumeInformationW(volume.c_str(), nullptr, 0, nullptr, nullptr, nullptr,
                                  filesystem.data(), static_cast<DWORD>(filesystem.size()))) {
            info.filesystem = Utf8(filesystem.data());
        }
        for (DWORD index = 0; index < extents->NumberOfDiskExtents; ++index) {
            by_disk[extents->Extents[index].DiskNumber].push_back(info);
        }
    } while (FindNextVolumeW(search, volume_name.data(),
                             static_cast<DWORD>(volume_name.size())));
    FindVolumeClose(search);
    return by_disk;
}

void DiscoverStorage(MachineSnapshot& snapshot, std::vector<std::string>& warnings) {
    const auto volumes = DiscoverVolumes();
    DeviceInfoSet devices(SetupDiGetClassDevsW(&kDiskInterfaceGuid, nullptr, nullptr,
                                                DIGCF_PRESENT | DIGCF_DEVICEINTERFACE));
    if (!devices.valid()) {
        warnings.emplace_back("Windows physical storage enumeration was unavailable");
        return;
    }
    snapshot.storage_discovery_available = true;
    for (DWORD index = 0;; ++index) {
        SP_DEVICE_INTERFACE_DATA interface_data{};
        interface_data.cbSize = sizeof(interface_data);
        if (!SetupDiEnumDeviceInterfaces(devices.get(), nullptr, &kDiskInterfaceGuid,
                                         index, &interface_data)) {
            if (GetLastError() != ERROR_NO_MORE_ITEMS) {
                warnings.emplace_back("Windows storage interface enumeration ended unexpectedly");
            }
            break;
        }
        DWORD required = 0;
        SetupDiGetDeviceInterfaceDetailW(devices.get(), &interface_data, nullptr, 0,
                                         &required, nullptr);
        if (required == 0) continue;
        std::vector<BYTE> detail_buffer(required);
        auto* detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(detail_buffer.data());
        detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
        SP_DEVINFO_DATA device_data{};
        device_data.cbSize = sizeof(device_data);
        if (!SetupDiGetDeviceInterfaceDetailW(devices.get(), &interface_data, detail,
                                              required, nullptr, &device_data)) {
            continue;
        }
        const std::string interface_path = Utf8(detail->DevicePath);
        const auto instance_id = QueryDeviceInstanceId(devices.get(), device_data);
        HANDLE raw_handle = CreateFileW(detail->DevicePath, GENERIC_READ,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, 0, nullptr);
        if (raw_handle == INVALID_HANDLE_VALUE) {
            raw_handle = CreateFileW(detail->DevicePath, 0,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     nullptr, OPEN_EXISTING, 0, nullptr);
        }
        Handle handle(raw_handle);
        if (!handle.valid()) continue;

        StorageDeviceInfo storage;
        const std::string identity_source = instance_id.value_or(interface_path);
        storage.persistent_id = "windows-device-sha256:" +
            core::Sha256Hex("SIDECAR-STORAGE-DEVICE-V1\n" + identity_source + "\n");
        storage.location_paths = QueryLocationPaths(devices.get(), device_data);
        bool used_ancestor_location = false;
        if (storage.location_paths.empty()) {
            storage.location_paths = QueryNearestLocationPaths(
                device_data.DevInst, used_ancestor_location);
        }
        storage.topology_confidence = storage.location_paths.empty()
                                          ? DiscoveryConfidence::Unknown
                                          : (used_ancestor_location
                                                 ? DiscoveryConfidence::Derived
                                                 : DiscoveryConfidence::DirectlyReported);
        storage.topology_source = storage.location_paths.empty()
                                      ? "Windows device interface"
                                      : (used_ancestor_location
                                             ? "CM_Get_Parent + DEVPKEY_Device_LocationPaths"
                                             : "DEVPKEY_Device_LocationPaths");

        STORAGE_PROPERTY_QUERY query{};
        query.PropertyId = StorageDeviceProperty;
        query.QueryType = PropertyStandardQuery;
        STORAGE_DESCRIPTOR_HEADER descriptor_header{};
        DWORD returned = 0;
        if (DeviceIoControl(handle.get(), IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
                            &descriptor_header, sizeof(descriptor_header), &returned, nullptr) &&
            descriptor_header.Size >= sizeof(STORAGE_DEVICE_DESCRIPTOR) &&
            descriptor_header.Size < 1024U * 1024U) {
            std::vector<BYTE> descriptor_buffer(descriptor_header.Size);
            if (DeviceIoControl(handle.get(), IOCTL_STORAGE_QUERY_PROPERTY,
                                &query, sizeof(query), descriptor_buffer.data(),
                                static_cast<DWORD>(descriptor_buffer.size()), &returned, nullptr)) {
                const auto* descriptor =
                    reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR*>(descriptor_buffer.data());
                const std::string vendor = DescriptorString(
                    descriptor, descriptor->VendorIdOffset, descriptor_buffer.size());
                const std::string product = DescriptorString(
                    descriptor, descriptor->ProductIdOffset, descriptor_buffer.size());
                storage.model = Trim(vendor + " " + product);
                storage.firmware = DescriptorString(
                    descriptor, descriptor->ProductRevisionOffset, descriptor_buffer.size());
                storage.bus_type = BusTypeName(descriptor->BusType);
            }
        }
        GET_LENGTH_INFORMATION length{};
        if (DeviceIoControl(handle.get(), IOCTL_DISK_GET_LENGTH_INFO, nullptr, 0,
                            &length, sizeof(length), &returned, nullptr)) {
            storage.capacity_bytes = static_cast<std::uint64_t>(length.Length.QuadPart);
        } else {
            std::array<BYTE, 1024> geometry_buffer{};
            if (DeviceIoControl(handle.get(), IOCTL_DISK_GET_DRIVE_GEOMETRY_EX,
                                nullptr, 0, geometry_buffer.data(),
                                static_cast<DWORD>(geometry_buffer.size()), &returned, nullptr)) {
                const auto* geometry =
                    reinterpret_cast<const DISK_GEOMETRY_EX*>(geometry_buffer.data());
                storage.capacity_bytes =
                    static_cast<std::uint64_t>(geometry->DiskSize.QuadPart);
            }
        }
        STORAGE_DEVICE_NUMBER number{};
        if (DeviceIoControl(handle.get(), IOCTL_STORAGE_GET_DEVICE_NUMBER, nullptr, 0,
                            &number, sizeof(number), &returned, nullptr)) {
            storage.physical_disk_number = number.DeviceNumber;
            const auto volume_match = volumes.find(number.DeviceNumber);
            if (volume_match != volumes.end()) storage.volumes = volume_match->second;
        }
        if (storage.model.empty()) storage.model = "Unknown physical disk";
        snapshot.storage_devices.push_back(std::move(storage));
    }
    std::sort(snapshot.storage_devices.begin(), snapshot.storage_devices.end(),
              [](const StorageDeviceInfo& left, const StorageDeviceInfo& right) {
                  return left.persistent_id < right.persistent_id;
              });
}

void AttachGpuLocationPaths(MachineSnapshot& snapshot) {
    DeviceInfoSet devices(SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr,
                                                DIGCF_PRESENT));
    if (!devices.valid()) return;
    for (DWORD index = 0;; ++index) {
        SP_DEVINFO_DATA device{};
        device.cbSize = sizeof(device);
        if (!SetupDiEnumDeviceInfo(devices.get(), index, &device)) break;
        const auto bus = QueryDeviceDword(devices.get(), device, DEVPKEY_Device_BusNumber);
        const auto address = QueryDeviceDword(devices.get(), device, DEVPKEY_Device_Address);
        if (!bus.has_value() || !address.has_value()) continue;
        const std::uint32_t device_number = *address >> 16U;
        auto match = std::find_if(snapshot.gpus.begin(), snapshot.gpus.end(),
                                  [&](const GpuInfo& gpu) {
                                      return gpu.pci_bus == bus && gpu.pci_device == device_number;
                                  });
        if (match != snapshot.gpus.end()) {
            match->location_paths = QueryLocationPaths(devices.get(), device);
        }
    }
}

void BuildTopology(MachineSnapshot& snapshot) {
    snapshot.topology.push_back(TopologyNode{
        "machine", std::nullopt, "MACHINE", "Local physical host",
        DiscoveryConfidence::DirectlyReported, "Sidecar discovery root", {}, {}});
    for (const auto& node : snapshot.cpu.numa_nodes) {
        snapshot.topology.push_back(TopologyNode{
            "numa:" + std::to_string(node.node_id), "machine", "NUMA_NODE",
            "NUMA Node " + std::to_string(node.node_id),
            DiscoveryConfidence::DirectlyReported,
            "GetLogicalProcessorInformationEx", {}, {}});
    }
    const std::string cpu_parent = snapshot.cpu.numa_nodes.size() == 1U
        ? "numa:" + std::to_string(snapshot.cpu.numa_nodes.front().node_id)
        : "machine";
    const std::uint32_t package_count = snapshot.cpu.package_count.value_or(1U);
    for (std::uint32_t package = 0; package < package_count; ++package) {
        snapshot.topology.push_back(TopologyNode{
            "cpu-package:" + std::to_string(package), cpu_parent, "CPU_PACKAGE",
            snapshot.cpu.model, DiscoveryConfidence::Derived,
            "GetLogicalProcessorInformationEx package relation", {}, {}});
    }
    for (std::size_t index = 0; index < snapshot.gpus.size(); ++index) {
        const auto& gpu = snapshot.gpus[index];
        std::vector<std::pair<std::string, std::string>> attributes;
        if (gpu.pci_bus.has_value()) attributes.emplace_back("pci_bus", std::to_string(*gpu.pci_bus));
        snapshot.topology.push_back(TopologyNode{
            "gpu:" + gpu.persistent_id, "machine", "GPU", gpu.model,
            DiscoveryConfidence::Derived,
            gpu.location_paths.empty()
                ? "CUDA/NVML PCI identity; physical parent unresolved"
                : "Windows PnP endpoint path; root/bridge parent unresolved",
            gpu.location_paths, std::move(attributes)});
    }
    for (const auto& storage : snapshot.storage_devices) {
        snapshot.topology.push_back(TopologyNode{
            "storage:" + storage.persistent_id, "machine", "STORAGE_DEVICE", storage.model,
            storage.topology_confidence, storage.topology_source,
            storage.location_paths,
            {{"bus_type", storage.bus_type}}});
    }
}

class WindowsHardwareDiscoveryProvider final : public IHardwareDiscoveryProvider {
public:
    RawDiscovery Discover() override {
        RawDiscovery discovery;
        DiscoverOperatingSystem(discovery.snapshot);
        DiscoverSmbios(discovery);
        DiscoverCpu(discovery.snapshot, discovery.warnings);
        DiscoverMemory(discovery.snapshot);
        discovery.identity_input.fallback_installation_id = ReadRegistryString(
            HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Cryptography", L"MachineGuid");
        DiscoverCudaDevices(discovery.snapshot, discovery.warnings);
        EnrichNvidiaDevicesWithNvml(discovery.snapshot, discovery.warnings);
        AttachGpuLocationPaths(discovery.snapshot);
        DiscoverStorage(discovery.snapshot, discovery.warnings);
        BuildTopology(discovery.snapshot);
        std::sort(discovery.snapshot.gpus.begin(), discovery.snapshot.gpus.end(),
                  [](const GpuInfo& left, const GpuInfo& right) {
                      return left.persistent_id < right.persistent_id;
                  });
        std::sort(discovery.snapshot.topology.begin(), discovery.snapshot.topology.end(),
                  [](const TopologyNode& left, const TopologyNode& right) {
                      return left.node_id < right.node_id;
                  });
        return discovery;
    }
};

}  // namespace

std::unique_ptr<IHardwareDiscoveryProvider> CreateNativeDiscoveryProvider() {
    return std::make_unique<WindowsHardwareDiscoveryProvider>();
}

}  // namespace sidecar::hardware
