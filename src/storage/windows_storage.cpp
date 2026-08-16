#ifdef _WIN32

#include "sidecar/storage/physics.hpp"

#include "sidecar/hardware/hardware.hpp"
#include "sidecar/memory/host_memory.hpp"

#define NOMINMAX
#include <Windows.h>
#include <winioctl.h>
#include <nvme.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cwctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sidecar::storage {
namespace {

class Handle final {
public:
    Handle() = default;
    explicit Handle(HANDLE value) : value_(value) {}
    ~Handle() { if (value_ && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : value_(other.value_) { other.value_ = nullptr; }
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            if (value_ && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_);
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }
    [[nodiscard]] HANDLE get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept {
        return value_ && value_ != INVALID_HANDLE_VALUE;
    }
private:
    HANDLE value_{nullptr};
};

std::uint64_t FileTimeValue(const FILETIME& value) {
    ULARGE_INTEGER integer{};
    integer.LowPart = value.dwLowDateTime;
    integer.HighPart = value.dwHighDateTime;
    return integer.QuadPart * 100ULL;
}

struct CpuTimes {
    std::uint64_t process_kernel{0}, process_user{0}, thread_kernel{0}, thread_user{0};
};

CpuTimes CaptureCpuTimes() {
    CpuTimes result;
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        result.process_kernel = FileTimeValue(kernel);
        result.process_user = FileTimeValue(user);
    }
    if (GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) {
        result.thread_kernel = FileTimeValue(kernel);
        result.thread_user = FileTimeValue(user);
    }
    return result;
}

std::uint64_t NowNs() {
    static const auto frequency = [] {
        LARGE_INTEGER value{};
        QueryPerformanceFrequency(&value);
        return static_cast<std::uint64_t>(value.QuadPart);
    }();
    LARGE_INTEGER value{};
    QueryPerformanceCounter(&value);
    return static_cast<std::uint64_t>((static_cast<long double>(value.QuadPart) *
                                       1000000000.0L) / frequency);
}

std::wstring Lower(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    return value;
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool ContainsInsensitive(std::string value, std::string selector) {
    return Lower(std::move(value)).find(Lower(std::move(selector))) != std::string::npos;
}

std::wstring PhysicalDrivePath(std::uint32_t number) {
    return L"\\\\.\\PhysicalDrive" + std::to_wstring(number);
}

Handle OpenPhysical(std::uint32_t number) {
    return Handle(CreateFileW(PhysicalDrivePath(number).c_str(), 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr));
}

AlignmentInfo QueryAlignment(const std::filesystem::path& mount,
                             std::uint32_t physical_number) {
    AlignmentInfo value;
    DWORD sectors_per_cluster = 0, bytes_per_sector = 0, free_clusters = 0, clusters = 0;
    if (GetDiskFreeSpaceW(mount.c_str(), &sectors_per_cluster, &bytes_per_sector,
                          &free_clusters, &clusters)) {
        value.logical_sector_bytes = bytes_per_sector;
        value.file_offset_alignment_bytes = bytes_per_sector;
        value.buffer_alignment_bytes = bytes_per_sector;
        value.source = "GetDiskFreeSpaceW";
    }
    auto disk = OpenPhysical(physical_number);
    if (disk) {
        STORAGE_PROPERTY_QUERY query{};
        query.PropertyId = StorageAccessAlignmentProperty;
        query.QueryType = PropertyStandardQuery;
        STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR descriptor{};
        DWORD returned = 0;
        if (DeviceIoControl(disk.get(), IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
                            &descriptor, sizeof(descriptor), &returned, nullptr) &&
            returned >= sizeof(descriptor)) {
            if (descriptor.BytesPerLogicalSector)
                value.logical_sector_bytes = descriptor.BytesPerLogicalSector;
            if (descriptor.BytesPerPhysicalSector)
                value.physical_sector_bytes = descriptor.BytesPerPhysicalSector;
            value.file_offset_alignment_bytes = value.logical_sector_bytes;
            value.buffer_alignment_bytes = std::max(value.logical_sector_bytes,
                                                     value.physical_sector_bytes);
            value.source = "IOCTL_STORAGE_QUERY_PROPERTY(StorageAccessAlignmentProperty)";
        }
    }
    if (!value.physical_sector_bytes) value.physical_sector_bytes = value.logical_sector_bytes;
    return value;
}

std::uint64_t Low64(const UCHAR value[16]) {
    std::uint64_t result = 0;
    for (unsigned index = 0; index < 8; ++index)
        result |= static_cast<std::uint64_t>(value[index]) << (index * 8U);
    return result;
}

HealthSnapshot QueryNvmeHealth(std::uint32_t physical_number, std::string phase) {
    HealthSnapshot result;
    result.phase = std::move(phase);
    auto disk = OpenPhysical(physical_number);
    if (!disk) {
        result.message = "CreateFile on resolved physical disk failed: " +
                         std::to_string(GetLastError());
        return result;
    }
    constexpr DWORD kProtocolSize = sizeof(NVME_HEALTH_INFO_LOG);
    constexpr DWORD kBufferSize = sizeof(STORAGE_PROPERTY_QUERY) +
                                  sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA) + kProtocolSize;
    std::vector<std::byte> buffer(kBufferSize);
    auto* query = reinterpret_cast<STORAGE_PROPERTY_QUERY*>(buffer.data());
    query->PropertyId = StorageDeviceProtocolSpecificProperty;
    query->QueryType = PropertyStandardQuery;
    auto* protocol = reinterpret_cast<STORAGE_PROTOCOL_SPECIFIC_DATA*>(
        query->AdditionalParameters);
    protocol->ProtocolType = ProtocolTypeNvme;
    protocol->DataType = NVMeDataTypeLogPage;
    protocol->ProtocolDataRequestValue = NVME_LOG_PAGE_HEALTH_INFO;
    protocol->ProtocolDataRequestSubValue = 0;
    protocol->ProtocolDataOffset = sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA);
    protocol->ProtocolDataLength = kProtocolSize;

    DWORD returned = 0;
    if (!DeviceIoControl(disk.get(), IOCTL_STORAGE_QUERY_PROPERTY, buffer.data(),
                         static_cast<DWORD>(buffer.size()), buffer.data(),
                         static_cast<DWORD>(buffer.size()), &returned, nullptr)) {
        result.message = "NVMe SMART/health protocol query unsupported or denied: " +
                         std::to_string(GetLastError());
        return result;
    }
    if (returned < sizeof(STORAGE_PROTOCOL_DATA_DESCRIPTOR)) {
        result.message = "NVMe health response was shorter than its descriptor";
        return result;
    }
    const auto* descriptor =
        reinterpret_cast<const STORAGE_PROTOCOL_DATA_DESCRIPTOR*>(buffer.data());
    const auto& data = descriptor->ProtocolSpecificData;
    const auto base = reinterpret_cast<const std::byte*>(&descriptor->ProtocolSpecificData);
    if (data.ProtocolDataLength < sizeof(NVME_HEALTH_INFO_LOG) ||
        base + data.ProtocolDataOffset + sizeof(NVME_HEALTH_INFO_LOG) >
            buffer.data() + returned) {
        result.message = "NVMe health response has invalid protocol data bounds";
        return result;
    }
    const auto* health = reinterpret_cast<const NVME_HEALTH_INFO_LOG*>(
        base + data.ProtocolDataOffset);
    result.critical_warning = health->CriticalWarning.AsUchar;
    const std::uint16_t kelvin = static_cast<std::uint16_t>(health->Temperature[0]) |
                                 (static_cast<std::uint16_t>(health->Temperature[1]) << 8U);
    if (kelvin) result.temperature_c = static_cast<double>(kelvin) - 273.15;
    result.available_spare_percent = health->AvailableSpare;
    result.available_spare_threshold_percent = health->AvailableSpareThreshold;
    result.percentage_used = health->PercentageUsed;
    result.data_units_read_low64 = Low64(health->DataUnitRead);
    result.data_units_written_low64 = Low64(health->DataUnitWritten);
    result.host_read_commands_low64 = Low64(health->HostReadCommands);
    result.host_write_commands_low64 = Low64(health->HostWrittenCommands);
    result.controller_busy_minutes_low64 = Low64(health->ControllerBusyTime);
    result.power_cycles_low64 = Low64(health->PowerCycle);
    result.power_on_hours_low64 = Low64(health->PowerOnHours);
    result.unsafe_shutdowns_low64 = Low64(health->UnsafeShutdowns);
    result.media_data_errors_low64 = Low64(health->MediaErrors);
    result.error_log_entries_low64 = Low64(health->ErrorInfoLogEntryCount);
    std::ostringstream raw;
    raw << "{\"bytes_returned\":" << returned
        << ",\"ioctl\":\"IOCTL_STORAGE_QUERY_PROPERTY\","
        << "\"log_page\":\"NVME_LOG_PAGE_HEALTH_INFO\","
        << "\"physical_disk_number\":" << physical_number
        << ",\"protocol_data_length\":" << data.ProtocolDataLength << '}';
    result.raw_evidence_json = raw.str();
    result.status = "SUCCESS";
    return result;
}

DatasetIdentity BaseDataset(const StorageTarget& target, std::uint64_t size) {
    DatasetIdentity result;
    result.path = target.dataset_path;
    result.size_bytes = size;
    result.identity_hash = DatasetIdentityHash(size);
    std::error_code error;
    result.exists = std::filesystem::is_regular_file(result.path, error);
    if (result.exists) {
        const auto observed = std::filesystem::file_size(result.path, error);
        if (error || observed != size) {
            result.message = error ? error.message() : "existing file has the wrong size";
        }
    }
    return result;
}

struct IoContext {
    OVERLAPPED overlapped{};
    std::uint32_t slot{0};
    std::uint64_t request_index{0};
    std::uint64_t offset{0};
    std::uint64_t submitted_ns{0};
    std::byte* buffer{nullptr};
    bool active{false};
};

BenchmarkResult RunMapped(const StorageTarget& target, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.config = config;
    result.cache_classification = "OS_CACHE_INFLUENCED";
    if (const auto issue = ValidateConfig(config, target, kDefaultDatasetBytes)) {
        result.status = RunStatus::InvalidConfiguration;
        result.message = *issue;
        return result;
    }
    Handle file(CreateFileW(target.dataset_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file) {
        result.status = RunStatus::IoError;
        result.message = "CreateFile failed: " + std::to_string(GetLastError());
        return result;
    }
    Handle mapping(CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr));
    if (!mapping) {
        result.status = RunStatus::IoError;
        result.message = "CreateFileMapping failed: " + std::to_string(GetLastError());
        return result;
    }
    const void* view = MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        result.status = RunStatus::IoError;
        result.message = "MapViewOfFile failed: " + std::to_string(GetLastError());
        return result;
    }
    auto& memory = memory::NativeHostMemoryProvider();
    memory::MemoryRegion region;
    auto allocated = config.destination == DestinationKind::CudaHostAlloc
        ? memory.AllocateCudaHost(config.block_bytes, region)
        : memory.AllocatePageable(config.block_bytes, region);
    if (!allocated.ok()) {
        UnmapViewOfFile(view);
        result.status = RunStatus::SkippedUnsupported;
        result.message = allocated.message;
        return result;
    }
    std::uint64_t checksum = 0;
    (void)memory.Touch(region, 0, checksum);
    const auto offsets = GenerateOffsets(config, kDefaultDatasetBytes);
    result.samples.reserve(offsets.size());
    const auto cpu_before = CaptureCpuTimes();
    const auto wall_start = NowNs();
    for (std::uint64_t index = 0; index < offsets.size(); ++index) {
        RequestSample sample;
        sample.request_index = index;
        sample.file_offset = offsets[index];
        sample.requested_bytes = config.block_bytes;
        sample.submission_reference_ns = NowNs() - wall_start;
        const auto started = NowNs();
        std::memcpy(region.data, static_cast<const std::byte*>(view) + offsets[index],
                    static_cast<std::size_t>(config.block_bytes));
        const auto completed = NowNs();
        sample.completion_latency_ns = completed - started;
        sample.completed_bytes = config.block_bytes;
        sample.status = RunStatus::Success;
        sample.completion_processing_ns = NowNs() - completed;
        result.samples.push_back(sample);
    }
    result.wall_time_ns = NowNs() - wall_start;
    const auto cpu_after = CaptureCpuTimes();
    result.cpu.process_kernel_ns = cpu_after.process_kernel - cpu_before.process_kernel;
    result.cpu.process_user_ns = cpu_after.process_user - cpu_before.process_user;
    result.cpu.completion_thread_kernel_ns = cpu_after.thread_kernel - cpu_before.thread_kernel;
    result.cpu.completion_thread_user_ns = cpu_after.thread_user - cpu_before.thread_user;
    result.verified = !offsets.empty() &&
        VerifyDatasetBytes(region.data, offsets.back(), config.block_bytes);
    result.status = result.verified ? RunStatus::Success : RunStatus::VerificationFailure;
    if (config.destination == DestinationKind::CudaHostAlloc) (void)memory.FreeCudaHost(region);
    else (void)memory.FreePageable(region);
    UnmapViewOfFile(view);
    FinalizeBenchmark(result);
    return result;
}

BenchmarkResult RunIocp(const StorageTarget& target, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.config = config;
    result.cache_classification = config.backend == BackendKind::Buffered
        ? "OS_CACHE_INFLUENCED" : "CACHE_BYPASSED";
    if (const auto issue = ValidateConfig(config, target, kDefaultDatasetBytes)) {
        result.status = RunStatus::InvalidConfiguration;
        result.message = *issue;
        return result;
    }
    DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;
    if (config.backend == BackendKind::OverlappedUnbuffered) flags |= FILE_FLAG_NO_BUFFERING;
    Handle file(CreateFileW(target.dataset_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, flags, nullptr));
    if (!file) {
        result.status = RunStatus::IoError;
        result.message = "CreateFile failed: " + std::to_string(GetLastError());
        return result;
    }
    Handle port(CreateIoCompletionPort(file.get(), nullptr, 0, 1));
    if (!port) {
        result.status = RunStatus::IoError;
        result.message = "CreateIoCompletionPort failed: " + std::to_string(GetLastError());
        return result;
    }
    if (config.block_bytes > std::numeric_limits<DWORD>::max()) {
        result.status = RunStatus::InvalidConfiguration;
        result.message = "Windows ReadFile length exceeds DWORD";
        return result;
    }
    const std::uint64_t allocation_bytes = config.block_bytes * config.queue_depth;
    auto& memory = memory::NativeHostMemoryProvider();
    memory::MemoryRegion region;
    auto allocated = config.destination == DestinationKind::CudaHostAlloc
        ? memory.AllocateCudaHost(allocation_bytes, region)
        : memory.AllocatePageable(allocation_bytes, region);
    if (!allocated.ok()) {
        result.status = RunStatus::SkippedUnsupported;
        result.message = allocated.message;
        return result;
    }
    auto free_region = [&] {
        if (config.destination == DestinationKind::CudaHostAlloc) (void)memory.FreeCudaHost(region);
        else (void)memory.FreePageable(region);
    };
    const auto alignment = target.alignment.buffer_alignment_bytes;
    if (alignment == 0 || reinterpret_cast<std::uintptr_t>(region.data) % alignment != 0) {
        result.status = RunStatus::InvalidConfiguration;
        result.message = "destination address violates unbuffered alignment";
        free_region();
        return result;
    }
    std::uint64_t touch_checksum = 0;
    if (!memory.Touch(region, 0, touch_checksum).ok()) {
        result.status = RunStatus::IoError;
        result.message = "destination pretouch failed";
        free_region();
        return result;
    }
    const auto offsets = GenerateOffsets(config, kDefaultDatasetBytes);
    result.samples.reserve(offsets.size());
    std::vector<IoContext> contexts(config.queue_depth);
    for (std::uint32_t slot = 0; slot < config.queue_depth; ++slot) {
        contexts[slot].slot = slot;
        contexts[slot].buffer = static_cast<std::byte*>(region.data) +
                                config.block_bytes * slot;
    }

    std::uint64_t next_request = 0, completed_count = 0, in_flight = 0;
    const auto cpu_before = CaptureCpuTimes();
    const auto wall_start = NowNs();
    auto submit = [&](IoContext& context, std::uint64_t request_index) {
        context.overlapped = {};
        context.request_index = request_index;
        context.offset = offsets[request_index];
        context.overlapped.Offset = static_cast<DWORD>(context.offset);
        context.overlapped.OffsetHigh = static_cast<DWORD>(context.offset >> 32U);
        const auto before = NowNs();
        context.submitted_ns = before;
        const BOOL immediate = ReadFile(file.get(), context.buffer,
            static_cast<DWORD>(config.block_bytes), nullptr, &context.overlapped);
        const DWORD error = immediate ? ERROR_SUCCESS : GetLastError();
        const auto after = NowNs();
        RequestSample sample;
        sample.request_index = request_index;
        sample.batch_id = request_index / config.queue_depth;
        sample.file_offset = context.offset;
        sample.requested_bytes = config.block_bytes;
        sample.submission_reference_ns = before - wall_start;
        sample.submission_cost_ns = after - before;
        result.samples.push_back(sample);
        if (!immediate && error != ERROR_IO_PENDING) {
            result.samples.back().status = RunStatus::IoError;
            result.samples.back().native_error = error;
            return false;
        }
        context.active = true;
        ++in_flight;
        return true;
    };

    bool submission_failed = false;
    while (next_request < offsets.size() && in_flight < config.queue_depth) {
        if (!submit(contexts[static_cast<std::size_t>(in_flight)], next_request++)) {
            submission_failed = true;
            break;
        }
    }
    result.status = submission_failed ? RunStatus::IoError : RunStatus::Success;
    if (submission_failed) result.message = "initial ReadFile submission failed";
    while (in_flight > 0 && result.status == RunStatus::Success) {
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* overlapped = nullptr;
        const auto wait_started = NowNs();
        const BOOL ok = GetQueuedCompletionStatus(port.get(), &bytes, &key, &overlapped,
                                                  static_cast<DWORD>(config.timeout_ms));
        const auto observed = NowNs();
        if (!overlapped) {
            result.status = GetLastError() == WAIT_TIMEOUT ? RunStatus::TimedOut : RunStatus::IoError;
            result.message = "GetQueuedCompletionStatus failed: " +
                             std::to_string(GetLastError());
            break;
        }
        auto* context = CONTAINING_RECORD(overlapped, IoContext, overlapped);
        context->active = false;
        --in_flight;
        ++completed_count;
        auto& sample = result.samples[static_cast<std::size_t>(context->request_index)];
        sample.completion_latency_ns = observed - context->submitted_ns;
        sample.completed_bytes = bytes;
        sample.native_error = ok ? 0U : GetLastError();
        sample.status = ok && bytes == config.block_bytes ? RunStatus::Success : RunStatus::IoError;
        sample.completion_processing_ns = NowNs() - observed;
        (void)key;
        (void)wait_started;
        if (sample.status != RunStatus::Success) {
            result.status = RunStatus::IoError;
            result.message = "partial or failed overlapped read";
            break;
        }
        if (next_request < offsets.size()) {
            if (!submit(*context, next_request++)) {
                result.status = RunStatus::IoError;
                result.message = "ReadFile submission failed";
                break;
            }
        }
    }
    if (in_flight) {
        CancelIoEx(file.get(), nullptr);
        const auto cancel_deadline = NowNs() + 5000000000ULL;
        while (in_flight && NowNs() < cancel_deadline) {
            DWORD bytes = 0; ULONG_PTR key = 0; OVERLAPPED* overlapped = nullptr;
            GetQueuedCompletionStatus(port.get(), &bytes, &key, &overlapped, 100);
            if (overlapped) --in_flight;
        }
    }
    result.wall_time_ns = NowNs() - wall_start;
    const auto cpu_after = CaptureCpuTimes();
    result.cpu.process_kernel_ns = cpu_after.process_kernel - cpu_before.process_kernel;
    result.cpu.process_user_ns = cpu_after.process_user - cpu_before.process_user;
    result.cpu.completion_thread_kernel_ns = cpu_after.thread_kernel - cpu_before.thread_kernel;
    result.cpu.completion_thread_user_ns = cpu_after.thread_user - cpu_before.thread_user;
    result.verified = completed_count == offsets.size();
    if (result.verified) {
        for (const auto& context : contexts) {
            if (context.request_index < offsets.size() &&
                !VerifyDatasetBytes(context.buffer, context.offset, config.block_bytes)) {
                result.verified = false;
                result.status = RunStatus::VerificationFailure;
                result.message = "post-timing destination verification failed";
                break;
            }
        }
    }
    free_region();
    FinalizeBenchmark(result);
    return result;
}

class WindowsStorageProvider final : public IStorageProvider {
public:
    StorageTarget ResolveTarget(const std::optional<std::string>& selector,
                                const std::optional<std::filesystem::path>& dataset_path) override {
        auto discovery = hardware::CreateNativeDiscoveryProvider();
        hardware::HardwareDiscoveryService service(*discovery);
        const auto report = service.Discover();
        std::vector<const hardware::StorageDeviceInfo*> candidates;
        for (const auto& device : report.snapshot.storage_devices) {
            if (Lower(device.bus_type) != "nvme" || !device.physical_disk_number ||
                device.volumes.empty()) continue;
            if (!selector || ContainsInsensitive(device.model, *selector) ||
                ContainsInsensitive(device.persistent_id, *selector)) {
                candidates.push_back(&device);
            }
        }
        if (!selector && candidates.size() > 1) {
            std::vector<const hardware::StorageDeviceInfo*> preferred;
            for (const auto* device : candidates)
                if (ContainsInsensitive(device->model, "990 pro")) preferred.push_back(device);
            if (preferred.size() == 1) candidates = std::move(preferred);
        }
        if (candidates.size() != 1)
            throw std::runtime_error("storage target resolution requires exactly one mounted NVMe; "
                                     "use --device with model or persistent id");
        const auto& device = *candidates.front();
        const auto volume = std::find_if(device.volumes.begin(), device.volumes.end(),
            [](const auto& candidate) { return !candidate.mount_points.empty(); });
        if (volume == device.volumes.end())
            throw std::runtime_error("resolved NVMe has no mounted volume");
        StorageTarget target;
        target.machine_hash = report.identity.machine_hash;
        target.persistent_id = device.persistent_id;
        target.model = device.model;
        target.firmware = device.firmware;
        target.bus_type = device.bus_type;
        target.capacity_bytes = device.capacity_bytes.value_or(0);
        target.physical_disk_number = *device.physical_disk_number;
        target.mount_point = std::filesystem::path(volume->mount_points.front());
        target.volume_name = volume->volume_name;
        target.location_paths = device.location_paths;
        target.mapping_confidence = hardware::ToString(device.topology_confidence);
        target.mapping_source = device.topology_source;
        target.dataset_path = dataset_path.value_or(target.mount_point /
            "SidecarData" / "sidecar-storage-dataset-v1-8gib.bin");
        const auto root = Lower(std::filesystem::absolute(target.mount_point).wstring());
        const auto candidate = Lower(std::filesystem::absolute(target.dataset_path).wstring());
        if (!candidate.starts_with(root))
            throw std::runtime_error("dataset path is not on the resolved target volume");
        target.alignment = QueryAlignment(target.mount_point, target.physical_disk_number);
        return target;
    }

    DatasetIdentity InspectDataset(const StorageTarget& target) override {
        return BaseDataset(target, kDefaultDatasetBytes);
    }

    DatasetIdentity CreateDataset(const StorageTarget& target, std::uint64_t bytes) override {
        auto current = BaseDataset(target, bytes);
        if (current.exists && current.message.empty()) {
            auto verified = VerifyDataset(target, false);
            if (verified.verified) {
                verified.message = "existing deterministic dataset reused";
                return verified;
            }
            throw std::runtime_error("existing dataset failed verification; refusing to overwrite");
        }
        std::error_code error;
        std::filesystem::create_directories(target.dataset_path.parent_path(), error);
        if (error) throw std::runtime_error("cannot create dataset directory: " + error.message());
        ULARGE_INTEGER available{}, total{}, free_total{};
        if (!GetDiskFreeSpaceExW(target.mount_point.c_str(), &available, &total, &free_total) ||
            available.QuadPart < bytes + 1024ULL * 1024ULL * 1024ULL)
            throw std::runtime_error("insufficient free space for dataset plus 1 GiB reserve");
        Handle file(CreateFileW(target.dataset_path.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                                nullptr));
        if (!file) throw std::runtime_error("CreateFile(CREATE_NEW) failed: " +
                                            std::to_string(GetLastError()));
        constexpr std::uint64_t kChunk = 8ULL * 1024ULL * 1024ULL;
        std::vector<std::byte> buffer(static_cast<std::size_t>(kChunk));
        for (std::uint64_t offset = 0; offset < bytes; offset += kChunk) {
            const auto count = std::min(kChunk, bytes - offset);
            FillDatasetBytes(buffer.data(), offset, count);
            DWORD written = 0;
            if (!WriteFile(file.get(), buffer.data(), static_cast<DWORD>(count), &written, nullptr) ||
                written != count)
                throw std::runtime_error("dataset write failed: " + std::to_string(GetLastError()));
        }
        if (!FlushFileBuffers(file.get()))
            throw std::runtime_error("FlushFileBuffers failed: " + std::to_string(GetLastError()));
        file = Handle();
        auto verified = VerifyDataset(target, false);
        verified.message = verified.verified ? "dataset created, flushed, and sampled successfully"
                                             : "dataset created but verification failed";
        return verified;
    }

    DatasetIdentity VerifyDataset(const StorageTarget& target, bool full) override {
        auto result = BaseDataset(target, kDefaultDatasetBytes);
        result.full_verification = full;
        if (!result.exists || !result.message.empty()) return result;
        Handle file(CreateFileW(target.dataset_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
        if (!file) {
            result.message = "dataset open failed: " + std::to_string(GetLastError());
            return result;
        }
        constexpr std::uint64_t kChunk = 8ULL * 1024ULL * 1024ULL;
        constexpr std::uint64_t kSample = 1024ULL * 1024ULL;
        std::vector<std::uint64_t> offsets;
        if (full) {
            for (std::uint64_t offset = 0; offset < result.size_bytes; offset += kChunk)
                offsets.push_back(offset);
        } else {
            offsets = {0, kSample, result.size_bytes / 7, result.size_bytes / 3,
                       result.size_bytes / 2, result.size_bytes * 2 / 3,
                       result.size_bytes - kSample * 2, result.size_bytes - kSample};
            for (auto& offset : offsets) offset = (offset / kSample) * kSample;
        }
        std::vector<std::byte> buffer(static_cast<std::size_t>(full ? kChunk : kSample));
        for (const auto offset : offsets) {
            const auto count = std::min<std::uint64_t>(buffer.size(), result.size_bytes - offset);
            LARGE_INTEGER position{}; position.QuadPart = static_cast<LONGLONG>(offset);
            if (!SetFilePointerEx(file.get(), position, nullptr, FILE_BEGIN)) {
                result.message = "dataset seek failed";
                return result;
            }
            DWORD read = 0;
            if (!ReadFile(file.get(), buffer.data(), static_cast<DWORD>(count), &read, nullptr) ||
                read != count || !VerifyDatasetBytes(buffer.data(), offset, count)) {
                result.message = "dataset content mismatch at offset " + std::to_string(offset);
                return result;
            }
            result.verified_bytes += count;
        }
        result.verified = true;
        result.message = full ? "full deterministic verification passed"
                              : "deterministic sampled verification passed";
        return result;
    }

    HealthSnapshot CaptureHealth(const StorageTarget& target, std::string phase) override {
        return QueryNvmeHealth(target.physical_disk_number, std::move(phase));
    }

    BenchmarkResult Run(const StorageTarget& target,
                        const BenchmarkConfig& config) override {
        if (config.backend == BackendKind::DirectStorage) {
            BenchmarkResult result;
            result.config = config;
            result.status = RunStatus::SkippedUnsupported;
            result.cache_classification = "DIRECTSTORAGE_UNAVAILABLE";
            result.message = "DirectStorage provider is optional and unavailable in this build";
            return result;
        }
        if (config.backend == BackendKind::MemoryMapped) return RunMapped(target, config);
        return RunIocp(target, config);
    }

    bool DirectStorageAvailable() const noexcept override { return false; }
};

class WindowsAsyncStorageReader final : public IAsyncStorageReader {
public:
    WindowsAsyncStorageReader(const StorageTarget& target, std::uint32_t maximum_outstanding)
        : maximum_outstanding_(std::max<std::uint32_t>(1, maximum_outstanding)),
          alignment_(std::max<std::uint32_t>(1, target.alignment.buffer_alignment_bytes)) {
        const DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN |
                            FILE_FLAG_NO_BUFFERING;
        file_ = Handle(CreateFileW(target.dataset_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, flags, nullptr));
        if (!file_) throw std::runtime_error("CreateFile(async pipeline) failed: " +
                                             std::to_string(GetLastError()));
        port_ = Handle(CreateIoCompletionPort(file_.get(), nullptr, 0, 1));
        if (!port_) throw std::runtime_error("CreateIoCompletionPort(async pipeline) failed: " +
                                             std::to_string(GetLastError()));
    }

    ~WindowsAsyncStorageReader() override { Cancel(); }

    ExternalReadCompletion Submit(const ExternalReadRequest& request) override {
        ExternalReadCompletion result;
        result.token = request.token;
        result.file_offset = request.file_offset;
        result.requested_bytes = request.bytes;
        if (!request.destination || !request.bytes ||
            request.bytes > std::numeric_limits<DWORD>::max() ||
            request.file_offset % alignment_ || request.bytes % alignment_ ||
            reinterpret_cast<std::uintptr_t>(request.destination) % alignment_ ||
            active_.size() >= maximum_outstanding_) {
            result.status = RunStatus::InvalidConfiguration;
            result.native_error = ERROR_INVALID_PARAMETER;
            return result;
        }
        auto context = std::make_unique<Context>();
        context->request = request;
        context->overlapped.Offset = static_cast<DWORD>(request.file_offset);
        context->overlapped.OffsetHigh = static_cast<DWORD>(request.file_offset >> 32U);
        context->submitted_ns = NowNs();
        const auto before = context->submitted_ns;
        const BOOL immediate = ReadFile(file_.get(), request.destination,
            static_cast<DWORD>(request.bytes), nullptr, &context->overlapped);
        const DWORD error = immediate ? ERROR_SUCCESS : GetLastError();
        const auto after = NowNs();
        result.submitted_ns = before;
        result.submission_cost_ns = after - before;
        if (!immediate && error != ERROR_IO_PENDING) {
            result.status = RunStatus::IoError;
            result.native_error = error;
            return result;
        }
        context->submission_cost_ns = result.submission_cost_ns;
        auto* key = &context->overlapped;
        active_.emplace(key, std::move(context));
        return result;
    }

    ExternalReadCompletion Wait(std::uint64_t timeout_ms) override {
        ExternalReadCompletion result;
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* overlapped = nullptr;
        const auto bounded_timeout = static_cast<DWORD>(std::min<std::uint64_t>(
            timeout_ms, std::numeric_limits<DWORD>::max()));
        const BOOL ok = GetQueuedCompletionStatus(port_.get(), &bytes, &key, &overlapped,
                                                   bounded_timeout);
        const auto completed = NowNs();
        if (!overlapped) {
            result.status = GetLastError() == WAIT_TIMEOUT ? RunStatus::TimedOut
                                                           : RunStatus::IoError;
            result.native_error = GetLastError();
            return result;
        }
        const auto found = active_.find(overlapped);
        if (found == active_.end()) {
            result.status = RunStatus::IoError;
            result.native_error = ERROR_INVALID_DATA;
            return result;
        }
        const auto& context = *found->second;
        result.token = context.request.token;
        result.file_offset = context.request.file_offset;
        result.requested_bytes = context.request.bytes;
        result.completed_bytes = bytes;
        result.submitted_ns = context.submitted_ns;
        result.completed_ns = completed;
        result.submission_cost_ns = context.submission_cost_ns;
        result.native_error = ok ? 0U : GetLastError();
        result.status = ok && bytes == context.request.bytes ? RunStatus::Success
                                                             : RunStatus::IoError;
        active_.erase(found);
        (void)key;
        return result;
    }

    void Cancel() noexcept override {
        if (!file_ || active_.empty()) return;
        CancelIoEx(file_.get(), nullptr);
        const auto deadline = NowNs() + 5'000'000'000ULL;
        while (!active_.empty() && NowNs() < deadline) {
            DWORD bytes = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* overlapped = nullptr;
            (void)GetQueuedCompletionStatus(port_.get(), &bytes, &key, &overlapped, 50);
            if (overlapped) active_.erase(overlapped);
        }
        active_.clear();
    }

    std::uint32_t Outstanding() const noexcept override {
        return static_cast<std::uint32_t>(active_.size());
    }

private:
    struct Context {
        OVERLAPPED overlapped{};
        ExternalReadRequest request;
        std::uint64_t submitted_ns{0};
        std::uint64_t submission_cost_ns{0};
    };

    std::uint32_t maximum_outstanding_{1};
    std::uint32_t alignment_{1};
    Handle file_;
    Handle port_;
    std::unordered_map<OVERLAPPED*, std::unique_ptr<Context>> active_;
};

}  // namespace

std::unique_ptr<IStorageProvider> CreateNativeStorageProvider() {
    return std::make_unique<WindowsStorageProvider>();
}

std::unique_ptr<IAsyncStorageReader> CreateNativeAsyncStorageReader(
    const StorageTarget& target, std::uint32_t maximum_outstanding) {
    return std::make_unique<WindowsAsyncStorageReader>(target, maximum_outstanding);
}

}  // namespace sidecar::storage

#endif
