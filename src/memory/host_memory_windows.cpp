#include "sidecar/memory/host_memory.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#define NOMINMAX
#include <Windows.h>
#include <Psapi.h>

#if SIDECAR_CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace sidecar::memory {
namespace {

ProviderResult UnsupportedCuda() {
    return {ResultStatus::SkippedUnsupported, 0,
            "CUDA host memory support is disabled in this build"};
}

#if SIDECAR_CUDA_ENABLED
ProviderResult CudaResult(cudaError_t result, ResultStatus default_status,
                          const char* operation) {
    if (result == cudaSuccess) return {};
    ResultStatus status = default_status;
    if (result == cudaErrorMemoryAllocation) status = ResultStatus::CudaOutOfMemory;
    return {status, static_cast<std::int64_t>(result),
            std::string(operation) + ": " + cudaGetErrorString(result)};
}
#endif

class WindowsHostMemoryProvider final : public IHostMemoryProvider {
public:
    WindowsHostMemoryProvider() {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        page_size_ = info.dwPageSize;
    }

    MemorySnapshot Snapshot() const override {
        MemorySnapshot snapshot;
        ULONGLONG installed_kib = 0;
        if (GetPhysicallyInstalledSystemMemory(&installed_kib))
            snapshot.installed_physical_bytes = installed_kib * 1024ULL;

        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) {
            snapshot.visible_physical_bytes = memory.ullTotalPhys;
            snapshot.available_physical_bytes = memory.ullAvailPhys;
            snapshot.memory_load_percent = memory.dwMemoryLoad;
        }

        PERFORMANCE_INFORMATION performance{};
        performance.cb = sizeof(performance);
        if (GetPerformanceInfo(&performance, sizeof(performance))) {
            snapshot.system_commit_bytes =
                static_cast<std::uint64_t>(performance.CommitTotal) * performance.PageSize;
            snapshot.system_commit_limit_bytes =
                static_cast<std::uint64_t>(performance.CommitLimit) * performance.PageSize;
        }

        PROCESS_MEMORY_COUNTERS_EX process{};
        process.cb = sizeof(process);
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process),
                                 sizeof(process))) {
            snapshot.process_working_set_bytes = process.WorkingSetSize;
            snapshot.process_private_bytes = process.PrivateUsage;
            snapshot.process_page_fault_count = process.PageFaultCount;
        }
        return snapshot;
    }

    std::uint64_t PageSize() const noexcept override { return page_size_; }
    bool SupportsCudaHostMemory() const noexcept override {
#if SIDECAR_CUDA_ENABLED
        return true;
#else
        return false;
#endif
    }

    ProviderResult WarmupCuda() override {
#if SIDECAR_CUDA_ENABLED
        return CudaResult(cudaFree(nullptr), ResultStatus::CudaHostMemoryError,
                          "cudaFree(nullptr) warmup");
#else
        return UnsupportedCuda();
#endif
    }

    ProviderResult AllocatePageable(std::uint64_t bytes, MemoryRegion& region) override {
        if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                      std::numeric_limits<SIZE_T>::max()))
            return {ResultStatus::OsAllocationFailure, ERROR_INVALID_PARAMETER,
                    "invalid VirtualAlloc size"};
        void* address = VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes),
                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!address)
            return {ResultStatus::OsAllocationFailure,
                    static_cast<std::int64_t>(GetLastError()), "VirtualAlloc failed"};
        region = {address, bytes};
        return {};
    }

    ProviderResult FreePageable(MemoryRegion& region) noexcept override {
        if (!region.data) return {};
        if (!VirtualFree(region.data, 0, MEM_RELEASE))
            return {ResultStatus::CleanupFailure,
                    static_cast<std::int64_t>(GetLastError()), "VirtualFree failed"};
        region = {};
        return {};
    }

    ProviderResult AllocateCudaHost(std::uint64_t bytes, MemoryRegion& region) override {
#if SIDECAR_CUDA_ENABLED
        if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            return {ResultStatus::CudaHostMemoryError, 0, "cudaHostAlloc size overflow"};
        void* address = nullptr;
        const auto result = cudaHostAlloc(&address, static_cast<std::size_t>(bytes),
                                          cudaHostAllocDefault);
        const auto classified = CudaResult(result, ResultStatus::CudaHostMemoryError,
                                           "cudaHostAlloc(default)");
        if (classified.ok()) region = {address, bytes};
        return classified;
#else
        (void)bytes; (void)region;
        return UnsupportedCuda();
#endif
    }

    ProviderResult FreeCudaHost(MemoryRegion& region) noexcept override {
#if SIDECAR_CUDA_ENABLED
        if (!region.data) return {};
        const auto result = CudaResult(cudaFreeHost(region.data), ResultStatus::CleanupFailure,
                                       "cudaFreeHost");
        if (result.ok()) region = {};
        return result;
#else
        (void)region;
        return UnsupportedCuda();
#endif
    }

    ProviderResult RegisterCudaHost(MemoryRegion& region) override {
#if SIDECAR_CUDA_ENABLED
        if (!region.data) return {ResultStatus::RegistrationFailure, 0, "empty region"};
        return CudaResult(cudaHostRegister(region.data,
                                           static_cast<std::size_t>(region.size_bytes),
                                           cudaHostRegisterDefault),
                          ResultStatus::RegistrationFailure,
                          "cudaHostRegister(default)");
#else
        (void)region;
        return UnsupportedCuda();
#endif
    }

    ProviderResult UnregisterCudaHost(MemoryRegion& region) noexcept override {
#if SIDECAR_CUDA_ENABLED
        if (!region.data) return {};
        return CudaResult(cudaHostUnregister(region.data), ResultStatus::CleanupFailure,
                          "cudaHostUnregister");
#else
        (void)region;
        return UnsupportedCuda();
#endif
    }

    ProviderResult Touch(MemoryRegion& region, std::uint64_t pass,
                         std::uint64_t& checksum) noexcept override {
        if (!region.data || region.size_bytes == 0)
            return {ResultStatus::VerificationFailure, 0, "cannot touch an empty region"};
        auto* bytes = static_cast<volatile std::uint8_t*>(region.data);
        std::uint64_t local = checksum ^ (pass * 0x9E3779B97F4A7C15ULL);
        for (std::uint64_t offset = 0; offset < region.size_bytes; offset += page_size_) {
            const auto value = static_cast<std::uint8_t>(bytes[offset] +
                static_cast<std::uint8_t>((offset / page_size_ + pass) | 1ULL));
            bytes[offset] = value;
            local = (local << 7) ^ (local >> 3) ^ value ^ offset;
        }
        const auto last = region.size_bytes - 1;
        bytes[last] = static_cast<std::uint8_t>(bytes[last] + pass + 1);
        local ^= bytes[last];
        checksum = local;
        return {};
    }

private:
    std::uint64_t page_size_{4096};
};

}  // namespace

IHostMemoryProvider& NativeHostMemoryProvider() {
    static WindowsHostMemoryProvider provider;
    return provider;
}

HostMemoryError::HostMemoryError(ProviderResult result)
    : std::runtime_error(result.message.empty() ? ToString(result.status) : result.message),
      result_(std::move(result)) {}

const ProviderResult& HostMemoryError::result() const noexcept { return result_; }

const char* ToString(MemoryMethod method) noexcept {
    switch (method) {
        case MemoryMethod::Pageable: return "PAGEABLE";
        case MemoryMethod::CudaHostAlloc: return "CUDA_HOST_ALLOC";
        case MemoryMethod::CudaHostRegister: return "CUDA_HOST_REGISTER";
    }
    return "PAGEABLE";
}

const char* ToString(RegistrationMode mode) noexcept {
    switch (mode) {
        case RegistrationMode::NotApplicable: return "NOT_APPLICABLE";
        case RegistrationMode::Cold: return "REGISTER_COLD";
        case RegistrationMode::Pretouched: return "REGISTER_PRETOUCHED";
    }
    return "NOT_APPLICABLE";
}

const char* ToString(ResultStatus status) noexcept {
    switch (status) {
        case ResultStatus::Success: return "SUCCESS";
        case ResultStatus::SkippedSafetyLimit: return "SKIPPED_SAFETY_LIMIT";
        case ResultStatus::SkippedUnsupported: return "SKIPPED_UNSUPPORTED";
        case ResultStatus::CudaOutOfMemory: return "CUDA_OUT_OF_MEMORY";
        case ResultStatus::CudaHostMemoryError: return "CUDA_HOST_MEMORY_ERROR";
        case ResultStatus::OsAllocationFailure: return "OS_ALLOCATION_FAILURE";
        case ResultStatus::RegistrationFailure: return "REGISTRATION_FAILURE";
        case ResultStatus::CleanupFailure: return "CLEANUP_FAILURE";
        case ResultStatus::VerificationFailure: return "VERIFICATION_FAILURE";
        case ResultStatus::AbortedPressureSignal: return "ABORTED_PRESSURE_SIGNAL";
        case ResultStatus::InternalError: return "INTERNAL_ERROR";
    }
    return "INTERNAL_ERROR";
}

PersistentPinnedArena::PersistentPinnedArena(IHostMemoryProvider& provider,
                                             ArenaBackend backend,
                                             std::uint64_t capacity_bytes,
                                             bool pretouch_registered)
    : provider_(&provider), backend_(backend) {
    ProviderResult result;
    if (backend_ == ArenaBackend::CudaHostAlloc) {
        result = provider_->AllocateCudaHost(capacity_bytes, region_);
    } else {
        result = provider_->AllocatePageable(capacity_bytes, region_);
        if (result.ok() && pretouch_registered) {
            std::uint64_t checksum = 0;
            result = provider_->Touch(region_, 0, checksum);
        }
        if (result.ok()) {
            result = provider_->RegisterCudaHost(region_);
            registered_ = result.ok();
        }
    }
    if (!result.ok()) {
        if (registered_) (void)provider_->UnregisterCudaHost(region_);
        if (region_.data) {
            if (backend_ == ArenaBackend::CudaHostAlloc)
                (void)provider_->FreeCudaHost(region_);
            else
                (void)provider_->FreePageable(region_);
        }
        throw HostMemoryError(std::move(result));
    }
}

PersistentPinnedArena::~PersistentPinnedArena() { (void)Close(); }

std::uint64_t PersistentPinnedArena::capacity() const noexcept { return region_.size_bytes; }

std::span<std::byte> PersistentPinnedArena::View(std::uint64_t offset,
                                                 std::uint64_t length) {
    if (offset > region_.size_bytes || length > region_.size_bytes - offset)
        throw std::out_of_range("persistent arena view exceeds capacity");
    return {static_cast<std::byte*>(region_.data) + offset,
            static_cast<std::size_t>(length)};
}

ProviderResult PersistentPinnedArena::Close() noexcept {
    if (!provider_ || !region_.data) return cleanup_result_;
    if (backend_ == ArenaBackend::RegisteredPageable && registered_) {
        cleanup_result_ = provider_->UnregisterCudaHost(region_);
        if (cleanup_result_.ok()) registered_ = false;
    }
    if (cleanup_result_.ok() && !registered_) {
        cleanup_result_ = backend_ == ArenaBackend::CudaHostAlloc
                              ? provider_->FreeCudaHost(region_)
                              : provider_->FreePageable(region_);
    }
    return cleanup_result_;
}

ProviderResult PersistentPinnedArena::cleanupResult() const { return cleanup_result_; }

}  // namespace sidecar::memory
