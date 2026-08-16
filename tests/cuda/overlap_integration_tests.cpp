#include "sidecar/cuda/overlap.hpp"

#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
using namespace sidecar::cuda;

void Expect(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RunBackend(HostMemoryClass backend) {
  auto &provider = NativeOverlapProvider();
  const auto prepared = provider.Prepare(0, backend, 2ULL << 20);
  Expect(prepared.ok(), "overlap backend preparation failed");
  for (const auto workload : {ComputeWorkload::SyntheticAlu,
                              ComputeWorkload::MemoryBound,
                              ComputeWorkload::Fp16Gemm}) {
    const auto profile = provider.Calibrate(workload, 1000, 2, 3);
    Expect(profile.validated && profile.status == OverlapStatus::Success,
           "compute workload calibration/validation failed");
    const auto c0 = provider.RunTopology(
        profile, TransferDirection::H2D, 2ULL << 20, SampleMode::ComputeOnly,
        InstrumentationMode::Full, 2'000'000, 250'000, 0);
    const auto t0 = provider.RunTopology(
        profile, TransferDirection::H2D, 2ULL << 20, SampleMode::TransferOnly,
        InstrumentationMode::Full, 2'000'000, 250'000, 0);
    std::atomic_bool external_gate_released{false};
    const auto concurrent = provider.RunTopology(
        profile, TransferDirection::H2D, 2ULL << 20, SampleMode::Concurrent,
        InstrumentationMode::Full, 2'000'000, 250'000, 0,
        &external_gate_released);
    Expect(c0.status == OverlapStatus::Success && c0.compute_ns > 0 &&
               c0.gate_valid,
           "matched C0 topology failed");
    Expect(t0.status == OverlapStatus::Success && t0.transfer_ns > 0 &&
               t0.gate_valid,
           "matched T0 topology failed");
    Expect(concurrent.status == OverlapStatus::Success &&
               concurrent.compute_ns > 0 && concurrent.transfer_ns > 0 &&
               concurrent.makespan_device_primary_ns > 0 &&
               concurrent.makespan_device_crosscheck_ns > 0 &&
               concurrent.gate_valid &&
               external_gate_released.load(std::memory_order_acquire),
           "common-gate concurrent topology failed");
    const auto validation =
        provider.Validate(profile, TransferDirection::H2D, 2ULL << 20);
    Expect(validation.ok(), "post-timing validation failed");
  }
  Expect(provider.Release().ok(), "overlap provider cleanup failed");
}
} // namespace

int main() {
  try {
    RunBackend(HostMemoryClass::PinnedHostAlloc);
    RunBackend(HostMemoryClass::PinnedRegistered);
    std::cout << "CUDA overlap integration tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA overlap integration failure: " << error.what() << '\n';
    (void)NativeOverlapProvider().Release();
    return 1;
  }
}
