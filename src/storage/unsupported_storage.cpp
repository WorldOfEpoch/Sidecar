#ifndef _WIN32

#include "sidecar/storage/physics.hpp"

#include <stdexcept>

namespace sidecar::storage {
namespace {
class UnsupportedStorageProvider final : public IStorageProvider {
public:
    StorageTarget ResolveTarget(const std::optional<std::string>&,
                                const std::optional<std::filesystem::path>&) override {
        throw std::runtime_error("native storage physics is Windows-first and unavailable");
    }
    DatasetIdentity InspectDataset(const StorageTarget&) override { return {}; }
    DatasetIdentity CreateDataset(const StorageTarget&, std::uint64_t) override { return {}; }
    DatasetIdentity VerifyDataset(const StorageTarget&, bool) override { return {}; }
    HealthSnapshot CaptureHealth(const StorageTarget&, std::string phase) override {
        HealthSnapshot value; value.phase = std::move(phase); return value;
    }
    BenchmarkResult Run(const StorageTarget&, const BenchmarkConfig& config) override {
        BenchmarkResult value; value.config = config;
        value.status = RunStatus::SkippedUnsupported; return value;
    }
    bool DirectStorageAvailable() const noexcept override { return false; }
};
class UnsupportedAsyncStorageReader final : public IAsyncStorageReader {
public:
    ExternalReadCompletion Submit(const ExternalReadRequest& request) override {
        ExternalReadCompletion value;
        value.token = request.token;
        value.status = RunStatus::SkippedUnsupported;
        return value;
    }
    ExternalReadCompletion Wait(std::uint64_t) override {
        ExternalReadCompletion value;
        value.status = RunStatus::SkippedUnsupported;
        return value;
    }
    void Cancel() noexcept override {}
    std::uint32_t Outstanding() const noexcept override { return 0; }
};
}
std::unique_ptr<IStorageProvider> CreateNativeStorageProvider() {
    return std::make_unique<UnsupportedStorageProvider>();
}
std::unique_ptr<IAsyncStorageReader> CreateNativeAsyncStorageReader(
    const StorageTarget&, std::uint32_t) {
    return std::make_unique<UnsupportedAsyncStorageReader>();
}
}
#endif
