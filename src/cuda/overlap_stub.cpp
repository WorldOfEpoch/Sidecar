#include "sidecar/cuda/overlap.hpp"

namespace sidecar::cuda {
namespace {
class UnsupportedOverlapProvider final : public IOverlapProvider {
public:
  bool SupportsCuda() const noexcept override { return false; }
  DeviceMemoryInfo InspectDevice(int) override { return {}; }
  TransferTelemetry ReadTelemetry(int) override { return {}; }
  TransferError Prepare(int, HostMemoryClass, std::uint64_t) override {
    return {TransferStatus::SkippedUnsupported, 0,
            "CUDA overlap laboratory is disabled"};
  }
  WorkloadProfile Calibrate(ComputeWorkload workload, double target,
                            std::uint32_t, std::uint32_t) override {
    WorkloadProfile profile;
    profile.workload = workload;
    profile.target_us = target;
    profile.status = OverlapStatus::SkippedUnsupported;
    profile.message = "CUDA overlap laboratory is disabled";
    return profile;
  }
  RawOverlapTiming RunTopology(const WorkloadProfile &, TransferDirection,
                               std::uint64_t, SampleMode, InstrumentationMode,
                               std::uint64_t, std::uint64_t,
                               std::uint32_t,
                               std::atomic_bool *external_gate_released) override {
    if (external_gate_released)
      external_gate_released->store(true, std::memory_order_release);
    if (external_gate_released) external_gate_released->notify_all();
    RawOverlapTiming timing;
    timing.status = OverlapStatus::SkippedUnsupported;
    timing.message = "CUDA overlap laboratory is disabled";
    return timing;
  }
  TransferError Validate(const WorkloadProfile &, TransferDirection,
                         std::uint64_t) override {
    return {TransferStatus::SkippedUnsupported, 0,
            "CUDA overlap laboratory is disabled"};
  }
  TransferError Release() noexcept override { return {}; }
};
} // namespace

IOverlapProvider &NativeOverlapProvider() {
  static UnsupportedOverlapProvider provider;
  return provider;
}
} // namespace sidecar::cuda
