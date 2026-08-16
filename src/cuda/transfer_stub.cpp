#include "sidecar/cuda/transfer.hpp"

namespace sidecar::cuda {
namespace {

class UnsupportedTransferProvider final : public ITransferProvider {
public:
  bool SupportsCuda() const noexcept override { return false; }
  DeviceMemoryInfo InspectDevice(int device_index) override {
    DeviceMemoryInfo info;
    info.device_index = device_index;
    return info;
  }
  TransferTelemetry ReadTelemetry(int) override { return {}; }
  TransferError Prepare(int, HostMemoryClass, std::uint64_t,
                        std::uint32_t) override {
    return Unsupported();
  }
  TransferError Warmup(TransferDirection, std::uint64_t) override {
    return Unsupported();
  }
  TransferSample RunCopy(const CopyRequest &, std::uint32_t) override {
    TransferSample sample;
    sample.status = TransferStatus::SkippedUnsupported;
    sample.message = "CUDA transfer support is disabled";
    return sample;
  }
  BidirectionalSample RunBidirectional(std::uint64_t, std::uint32_t) override {
    BidirectionalSample sample;
    sample.status = TransferStatus::SkippedUnsupported;
    return sample;
  }
  TransferError Verify(TransferDirection, std::uint64_t,
                       std::uint32_t) override {
    return Unsupported();
  }
  TransferError Release() noexcept override { return {}; }

private:
  static TransferError Unsupported() {
    return {TransferStatus::SkippedUnsupported, 0,
            "CUDA transfer support is disabled in this build"};
  }
};

} // namespace

ITransferProvider &NativeTransferProvider() {
  static UnsupportedTransferProvider provider;
  return provider;
}

} // namespace sidecar::cuda
