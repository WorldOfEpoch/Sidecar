#include "sidecar/memory/host_memory.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace sidecar::memory {
namespace {

class PortableFallbackProvider final : public IHostMemoryProvider {
public:
    MemorySnapshot Snapshot() const override { return {}; }
    std::uint64_t PageSize() const noexcept override { return 4096; }
    bool SupportsCudaHostMemory() const noexcept override { return false; }
    ProviderResult WarmupCuda() override {
        return {ResultStatus::SkippedUnsupported, 0, "CUDA host memory unavailable"};
    }
    ProviderResult AllocatePageable(std::uint64_t bytes, MemoryRegion& region) override {
        region.data = std::malloc(static_cast<std::size_t>(bytes));
        if (!region.data) return {ResultStatus::OsAllocationFailure, 0, "malloc failed"};
        region.size_bytes = bytes; return {};
    }
    ProviderResult FreePageable(MemoryRegion& region) noexcept override {
        std::free(region.data); region = {}; return {};
    }
    ProviderResult AllocateCudaHost(std::uint64_t, MemoryRegion&) override {
        return WarmupCuda();
    }
    ProviderResult FreeCudaHost(MemoryRegion&) noexcept override { return WarmupCuda(); }
    ProviderResult RegisterCudaHost(MemoryRegion&) override { return WarmupCuda(); }
    ProviderResult UnregisterCudaHost(MemoryRegion&) noexcept override { return WarmupCuda(); }
    ProviderResult Touch(MemoryRegion& region, std::uint64_t pass,
                         std::uint64_t& checksum) noexcept override {
        if (!region.data) return {ResultStatus::VerificationFailure, 0, "empty region"};
        auto* bytes = static_cast<volatile unsigned char*>(region.data);
        for (std::uint64_t offset = 0; offset < region.size_bytes; offset += 4096) {
            bytes[offset] = static_cast<unsigned char>(pass + offset / 4096);
            checksum ^= bytes[offset];
        }
        return {};
    }
};

}  // namespace

IHostMemoryProvider& NativeHostMemoryProvider() {
    static PortableFallbackProvider provider;
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

PersistentPinnedArena::PersistentPinnedArena(IHostMemoryProvider&, ArenaBackend,
                                             std::uint64_t, bool) {
    throw HostMemoryError({ResultStatus::SkippedUnsupported, 0,
                           "persistent pinned arenas are unsupported"});
}
PersistentPinnedArena::~PersistentPinnedArena() = default;
std::uint64_t PersistentPinnedArena::capacity() const noexcept { return 0; }
std::span<std::byte> PersistentPinnedArena::View(std::uint64_t, std::uint64_t) { return {}; }
ProviderResult PersistentPinnedArena::Close() noexcept { return {}; }
ProviderResult PersistentPinnedArena::cleanupResult() const { return {}; }

}  // namespace sidecar::memory
