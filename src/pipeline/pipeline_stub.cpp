#include "sidecar/pipeline/physics.hpp"

namespace sidecar::pipeline {
namespace {
class UnsupportedPipelineProvider final : public IPipelineProvider {
public:
    bool SupportsCuda() const noexcept override { return false; }
    HostCopyResult RunHostCopy(const HostCopyConfiguration& configuration) override {
        HostCopyResult result;
        result.configuration = configuration;
        result.status = PipelineStatus::SkippedUnsupported;
        result.message = "CUDA host memory is disabled; pageable-to-pinned measurement skipped";
        return result;
    }
    PipelineResult Run(const storage::StorageTarget&,
                       const PipelineConfiguration& configuration) override {
        PipelineResult result;
        result.configuration = configuration;
        result.status = PipelineStatus::SkippedUnsupported;
        result.message = "CUDA pipeline support is disabled in this build";
        return result;
    }
};
}
std::unique_ptr<IPipelineProvider> CreateNativePipelineProvider() {
    return std::make_unique<UnsupportedPipelineProvider>();
}
}
