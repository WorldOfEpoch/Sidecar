#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace sidecar::memory {

enum class MemoryMethod { Pageable, CudaHostAlloc, CudaHostRegister };
enum class RegistrationMode { NotApplicable, Cold, Pretouched };
enum class ResultStatus {
    Success,
    SkippedSafetyLimit,
    SkippedUnsupported,
    CudaOutOfMemory,
    CudaHostMemoryError,
    OsAllocationFailure,
    RegistrationFailure,
    CleanupFailure,
    VerificationFailure,
    AbortedPressureSignal,
    InternalError,
};

struct MemorySnapshot {
    std::optional<std::uint64_t> installed_physical_bytes;
    std::optional<std::uint64_t> visible_physical_bytes;
    std::optional<std::uint64_t> available_physical_bytes;
    std::optional<std::uint64_t> system_commit_bytes;
    std::optional<std::uint64_t> system_commit_limit_bytes;
    std::optional<std::uint64_t> process_working_set_bytes;
    std::optional<std::uint64_t> process_private_bytes;
    std::optional<std::uint64_t> process_page_fault_count;
    std::optional<std::uint32_t> memory_load_percent;
};

struct ProviderResult {
    ResultStatus status{ResultStatus::Success};
    std::int64_t native_error{0};
    std::string message;
    [[nodiscard]] bool ok() const noexcept { return status == ResultStatus::Success; }
};

class HostMemoryError final : public std::runtime_error {
public:
    explicit HostMemoryError(ProviderResult result);
    [[nodiscard]] const ProviderResult& result() const noexcept;

private:
    ProviderResult result_;
};

struct MemoryRegion {
    void* data{nullptr};
    std::uint64_t size_bytes{0};
};

class IHostMemoryProvider {
public:
    virtual ~IHostMemoryProvider() = default;
    [[nodiscard]] virtual MemorySnapshot Snapshot() const = 0;
    [[nodiscard]] virtual std::uint64_t PageSize() const noexcept = 0;
    [[nodiscard]] virtual bool SupportsCudaHostMemory() const noexcept = 0;
    virtual ProviderResult WarmupCuda() = 0;
    virtual ProviderResult AllocatePageable(std::uint64_t bytes, MemoryRegion& region) = 0;
    virtual ProviderResult FreePageable(MemoryRegion& region) noexcept = 0;
    virtual ProviderResult AllocateCudaHost(std::uint64_t bytes, MemoryRegion& region) = 0;
    virtual ProviderResult FreeCudaHost(MemoryRegion& region) noexcept = 0;
    virtual ProviderResult RegisterCudaHost(MemoryRegion& region) = 0;
    virtual ProviderResult UnregisterCudaHost(MemoryRegion& region) noexcept = 0;
    virtual ProviderResult Touch(MemoryRegion& region, std::uint64_t pass,
                                 std::uint64_t& checksum) noexcept = 0;
};

[[nodiscard]] IHostMemoryProvider& NativeHostMemoryProvider();
[[nodiscard]] const char* ToString(MemoryMethod method) noexcept;
[[nodiscard]] const char* ToString(RegistrationMode mode) noexcept;
[[nodiscard]] const char* ToString(ResultStatus status) noexcept;

enum class ArenaBackend { CudaHostAlloc, RegisteredPageable };

class PersistentPinnedArena final {
public:
    PersistentPinnedArena(IHostMemoryProvider& provider,
                          ArenaBackend backend,
                          std::uint64_t capacity_bytes,
                          bool pretouch_registered = true);
    ~PersistentPinnedArena();
    PersistentPinnedArena(const PersistentPinnedArena&) = delete;
    PersistentPinnedArena& operator=(const PersistentPinnedArena&) = delete;
    PersistentPinnedArena(PersistentPinnedArena&&) = delete;
    PersistentPinnedArena& operator=(PersistentPinnedArena&&) = delete;

    [[nodiscard]] std::uint64_t capacity() const noexcept;
    [[nodiscard]] std::span<std::byte> View(std::uint64_t offset, std::uint64_t length);
    ProviderResult Close() noexcept;
    [[nodiscard]] ProviderResult cleanupResult() const;

private:
    IHostMemoryProvider* provider_{nullptr};
    ArenaBackend backend_{ArenaBackend::CudaHostAlloc};
    MemoryRegion region_;
    bool registered_{false};
    ProviderResult cleanup_result_;
};

}  // namespace sidecar::memory
