#include "sidecar/cuda/transfer.hpp"
#include <iostream>

int main() {
  using namespace sidecar::cuda;
  auto &provider = NativeTransferProvider();
  if (!provider.SupportsCuda()) {
    std::cout << "[SKIP] CUDA unavailable\n";
    return 0;
  }
  constexpr std::uint64_t bytes = 4ULL * 1024 * 1024;
  for (auto memory :
       {HostMemoryClass::PageablePretouched, HostMemoryClass::PinnedHostAlloc,
        HostMemoryClass::PinnedRegistered}) {
    auto prepared = provider.Prepare(0, memory, bytes, 1);
    if (!prepared.ok())
      return 1;
    for (auto direction : {TransferDirection::H2D, TransferDirection::D2H}) {
      if (!provider.Warmup(direction, bytes).ok())
        return 2;
      auto sample = provider.RunCopy(
          {direction, CopyApiMode::Asynchronous, bytes, 1, false, false}, 0);
      if (sample.status != TransferStatus::Success ||
          sample.device_duration_ns == 0 ||
          !provider.Verify(direction, bytes).ok())
        return 3;
    }
    if (!provider.Release().ok())
      return 4;
  }
  if (!provider.Prepare(0, HostMemoryClass::PinnedHostAlloc, bytes, 2).ok())
    return 5;
  auto bidi = provider.RunBidirectional(bytes, 0);
  if (bidi.status != TransferStatus::Success || bidi.h2d_device_ns == 0 ||
      bidi.d2h_device_ns == 0)
    return 6;
  if (!provider.Verify(TransferDirection::H2D, bytes, 0).ok() ||
      !provider.Verify(TransferDirection::D2H, bytes, 1).ok())
    return 7;
  if (!provider.Release().ok())
    return 8;
  std::cout << "[PASS] real CUDA transfer integration\n";
  return 0;
}
