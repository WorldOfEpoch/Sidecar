#include "sidecar/llama/observation.hpp"

#define NOMINMAX
#include <Windows.h>
#include <winioctl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace sidecar::llama {
namespace {

class Handle final {
public:
    explicit Handle(HANDLE value) : value_(value) {}
    ~Handle() { if (value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    HANDLE get() const noexcept { return value_; }
private:
    HANDLE value_;
};

std::runtime_error WindowsError(std::string_view operation) {
    return std::runtime_error(std::string(operation) + " failed with Windows error " +
                              std::to_string(GetLastError()));
}

std::string Narrow(const std::wstring& value) {
    if (value.empty()) return {};
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                        result.data(), bytes, nullptr, nullptr);
    return result;
}

std::string Trim(std::string value) {
    const auto non_space = [](unsigned char character) { return !std::isspace(character); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), non_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), non_space).base(), value.end());
    return value;
}

std::string BusTypeName(STORAGE_BUS_TYPE type) {
    switch (type) {
        case BusTypeScsi: return "SCSI";
        case BusTypeAtapi: return "ATAPI";
        case BusTypeAta: return "ATA";
        case BusType1394: return "IEEE1394";
        case BusTypeSsa: return "SSA";
        case BusTypeFibre: return "FIBRE_CHANNEL";
        case BusTypeUsb: return "USB";
        case BusTypeRAID: return "RAID";
        case BusTypeiScsi: return "ISCSI";
        case BusTypeSas: return "SAS";
        case BusTypeSata: return "SATA";
        case BusTypeSd: return "SD";
        case BusTypeMmc: return "MMC";
        case BusTypeVirtual: return "VIRTUAL";
        case BusTypeFileBackedVirtual: return "FILE_BACKED_VIRTUAL";
        case BusTypeSpaces: return "STORAGE_SPACES";
        case BusTypeNvme: return "NVME";
        default: return "UNKNOWN";
    }
}

}  // namespace

StorageProvenance ResolveModelStorage(const std::filesystem::path& path,
                                      std::string storage_role,
                                      std::string copy_relationship) {
    StorageProvenance result;
    result.requested_path = std::filesystem::absolute(path);
    result.storage_role = std::move(storage_role);
    result.copy_relationship = std::move(copy_relationship);
    Handle file(CreateFileW(result.requested_path.c_str(), 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) throw WindowsError("open model for storage provenance");

    std::vector<wchar_t> final_buffer(32768);
    const DWORD final_count = GetFinalPathNameByHandleW(file.get(), final_buffer.data(),
        static_cast<DWORD>(final_buffer.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (final_count == 0 || final_count >= final_buffer.size())
        throw WindowsError("resolve final model path");
    std::wstring final_path(final_buffer.data(), final_count);
    result.resolved_path = final_path;
    std::wstring local_path = final_path.starts_with(L"\\\\?\\")
        ? final_path.substr(4) : final_path;

    std::array<wchar_t, MAX_PATH + 1> volume_root{};
    if (!GetVolumePathNameW(local_path.c_str(), volume_root.data(),
                            static_cast<DWORD>(volume_root.size())))
        throw WindowsError("resolve model volume root");
    result.volume_path = Narrow(volume_root.data());
    std::array<wchar_t, MAX_PATH + 1> volume_name{};
    if (GetVolumeNameForVolumeMountPointW(volume_root.data(), volume_name.data(),
                                         static_cast<DWORD>(volume_name.size())))
        result.volume_unique_id = Narrow(volume_name.data());

    std::wstring volume_device;
    if (std::wcslen(volume_root.data()) >= 2 && volume_root[1] == L':') {
        volume_device = L"\\\\.\\";
        volume_device.push_back(volume_root[0]);
        volume_device.push_back(L':');
    } else {
        throw std::runtime_error("model volume is not represented by a drive-letter mount");
    }
    Handle volume(CreateFileW(volume_device.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING, 0, nullptr));
    if (volume.get() == INVALID_HANDLE_VALUE) throw WindowsError("open model volume");
    STORAGE_DEVICE_NUMBER number{};
    DWORD returned = 0;
    if (!DeviceIoControl(volume.get(), IOCTL_STORAGE_GET_DEVICE_NUMBER, nullptr, 0,
                         &number, sizeof(number), &returned, nullptr))
        throw WindowsError("map model volume to physical disk");
    result.physical_disk_number = number.DeviceNumber;
    result.physical_device = "\\\\.\\PhysicalDrive" + std::to_string(number.DeviceNumber);

    const std::wstring physical_path = L"\\\\.\\PhysicalDrive" +
        std::to_wstring(number.DeviceNumber);
    Handle disk(CreateFileW(physical_path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING, 0, nullptr));
    if (disk.get() == INVALID_HANDLE_VALUE) throw WindowsError("open model physical disk");
    STORAGE_PROPERTY_QUERY query{};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;
    std::array<std::uint8_t, 4096> descriptor_buffer{};
    if (!DeviceIoControl(disk.get(), IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
                         descriptor_buffer.data(), static_cast<DWORD>(descriptor_buffer.size()),
                         &returned, nullptr))
        throw WindowsError("query model physical disk descriptor");
    const auto* descriptor = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR*>(
        descriptor_buffer.data());
    auto string_at = [&](DWORD offset) {
        if (offset == 0 || offset >= returned) return std::string{};
        const auto* begin = reinterpret_cast<const char*>(descriptor_buffer.data() + offset);
        const auto* end = begin;
        const auto* limit = reinterpret_cast<const char*>(descriptor_buffer.data() + returned);
        while (end < limit && *end != '\0') ++end;
        return Trim(std::string(begin, end));
    };
    const std::string vendor = string_at(descriptor->VendorIdOffset);
    const std::string product = string_at(descriptor->ProductIdOffset);
    result.device_model = Trim(vendor + (vendor.empty() || product.empty() ? "" : " ") + product);
    if (result.device_model.empty()) result.device_model = "UNKNOWN";
    result.bus_type = BusTypeName(descriptor->BusType);
    std::string lower = result.device_model;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    result.is_samsung_990_pro = lower.find("samsung") != std::string::npos &&
                                lower.find("990 pro") != std::string::npos;
    result.is_usb_external = descriptor->BusType == BusTypeUsb;
    return result;
}

}  // namespace sidecar::llama
