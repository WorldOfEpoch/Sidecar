#include "sidecar/cuda/transfer.hpp"

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace sidecar::cuda {
namespace {

std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

TransferError
CudaResult(cudaError_t value, const char *operation,
           TransferStatus default_status = TransferStatus::CudaError) {
  if (value == cudaSuccess)
    return {};
  const auto status = value == cudaErrorMemoryAllocation
                          ? TransferStatus::CudaOutOfMemory
                          : default_status;
  return {status, static_cast<std::int64_t>(value),
          std::string(operation) + ": " + cudaGetErrorString(value)};
}

std::uint8_t PatternByte(std::uint64_t offset, std::uint32_t buffer) {
  return static_cast<std::uint8_t>(
      (offset * 1315423911ULL + buffer * 17ULL + 0x5A) & 0xFF);
}

template <typename Function>
Function LoadFunction(HMODULE module, const char *name) {
  return reinterpret_cast<Function>(GetProcAddress(module, name));
}

TransferTelemetry ReadNvmlTelemetry(int device_index) {
  TransferTelemetry result;
  HMODULE module = LoadLibraryW(L"nvml.dll");
  if (!module)
    module =
        LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
  if (!module)
    return result;
  using InitFn = nvmlReturn_t (*)();
  using ShutdownFn = nvmlReturn_t (*)();
  using HandleFn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
  using UIntFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
  using TempFn =
      nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
  using ClockFn =
      nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
  const auto init = LoadFunction<InitFn>(module, "nvmlInit_v2");
  const auto shutdown = LoadFunction<ShutdownFn>(module, "nvmlShutdown");
  const auto handle_by_index =
      LoadFunction<HandleFn>(module, "nvmlDeviceGetHandleByIndex_v2");
  if (!init || !shutdown || !handle_by_index || init() != NVML_SUCCESS) {
    FreeLibrary(module);
    return result;
  }
  nvmlDevice_t handle = nullptr;
  if (handle_by_index(static_cast<unsigned int>(device_index), &handle) !=
      NVML_SUCCESS) {
    shutdown();
    FreeLibrary(module);
    return result;
  }
  result.available = true;
  const auto read_uint = [&](const char *name, auto &destination) {
    const auto function = LoadFunction<UIntFn>(module, name);
    unsigned int value = 0;
    if (function && function(handle, &value) == NVML_SUCCESS)
      destination = value;
  };
  read_uint("nvmlDeviceGetCurrPcieLinkGeneration", result.pcie_generation);
  read_uint("nvmlDeviceGetCurrPcieLinkWidth", result.pcie_width);
  unsigned int value = 0;
  const auto temperature =
      LoadFunction<TempFn>(module, "nvmlDeviceGetTemperature");
  if (temperature &&
      temperature(handle, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS)
    result.temperature_c = value;
  const auto clock = LoadFunction<ClockFn>(module, "nvmlDeviceGetClockInfo");
  if (clock && clock(handle, NVML_CLOCK_GRAPHICS, &value) == NVML_SUCCESS)
    result.graphics_clock_mhz = value;
  if (clock && clock(handle, NVML_CLOCK_MEM, &value) == NVML_SUCCESS)
    result.memory_clock_mhz = value;
  const auto power = LoadFunction<UIntFn>(module, "nvmlDeviceGetPowerUsage");
  if (power && power(handle, &value) == NVML_SUCCESS)
    result.power_watts = static_cast<double>(value) / 1000.0;
  const auto power_limit =
      LoadFunction<UIntFn>(module, "nvmlDeviceGetPowerManagementLimit");
  if (power_limit && power_limit(handle, &value) == NVML_SUCCESS)
    result.power_limit_watts = static_cast<double>(value) / 1000.0;
  shutdown();
  FreeLibrary(module);
  return result;
}

class NativeCudaTransferProvider final : public ITransferProvider {
public:
  ~NativeCudaTransferProvider() override { (void)Release(); }
  bool SupportsCuda() const noexcept override { return true; }

  DeviceMemoryInfo InspectDevice(int device_index) override {
    DeviceMemoryInfo result;
    result.device_index = device_index;
    if (cudaSetDevice(device_index) != cudaSuccess) {
      (void)cudaGetLastError();
      return result;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess) {
      (void)cudaGetLastError();
      return result;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      return result;
    }
    result.cuda_supported = true;
    result.name = properties.name;
    result.free_bytes = free_bytes;
    result.total_bytes = total_bytes;
    result.async_engine_count =
        static_cast<std::uint32_t>(properties.asyncEngineCount);
    return result;
  }

  TransferTelemetry ReadTelemetry(int device_index) override {
    return ReadNvmlTelemetry(device_index);
  }

  TransferError Prepare(int device_index, HostMemoryClass memory_class,
                        std::uint64_t capacity_bytes,
                        std::uint32_t buffer_count) override {
    const auto prior = Release();
    if (!prior.ok())
      return prior;
    if (capacity_bytes == 0 || buffer_count == 0 ||
        capacity_bytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      return {TransferStatus::HostAllocationFailure, ERROR_INVALID_PARAMETER,
              "invalid transfer buffer capacity"};
    auto error = CudaResult(cudaSetDevice(device_index), "cudaSetDevice");
    if (!error.ok())
      return error;
    device_index_ = device_index;
    memory_class_ = memory_class;
    capacity_bytes_ = capacity_bytes;
    host_.resize(buffer_count, nullptr);
    device_.resize(buffer_count, nullptr);
    registered_.resize(buffer_count, false);
    for (std::uint32_t index = 0; index < buffer_count; ++index) {
      if (memory_class == HostMemoryClass::PinnedHostAlloc) {
        error =
            CudaResult(cudaHostAlloc(&host_[index],
                                     static_cast<std::size_t>(capacity_bytes),
                                     cudaHostAllocDefault),
                       "cudaHostAlloc", TransferStatus::HostAllocationFailure);
      } else {
        host_[index] =
            VirtualAlloc(nullptr, static_cast<SIZE_T>(capacity_bytes),
                         MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!host_[index])
          error = {TransferStatus::HostAllocationFailure,
                   static_cast<std::int64_t>(GetLastError()),
                   "VirtualAlloc failed"};
      }
      if (!error.ok()) {
        (void)Release();
        return error;
      }
      auto *bytes = static_cast<std::uint8_t *>(host_[index]);
      for (std::uint64_t offset = 0; offset < capacity_bytes; ++offset)
        bytes[offset] = PatternByte(offset, index);
      if (memory_class == HostMemoryClass::PinnedRegistered) {
        error = CudaResult(
            cudaHostRegister(host_[index],
                             static_cast<std::size_t>(capacity_bytes),
                             cudaHostRegisterDefault),
            "cudaHostRegister", TransferStatus::HostAllocationFailure);
        if (!error.ok()) {
          (void)Release();
          return error;
        }
        registered_[index] = true;
      }
      error = CudaResult(
          cudaMalloc(&device_[index], static_cast<std::size_t>(capacity_bytes)),
          "cudaMalloc");
      if (!error.ok()) {
        (void)Release();
        return error;
      }
      error = CudaResult(cudaMemcpy(device_[index], host_[index],
                                    static_cast<std::size_t>(capacity_bytes),
                                    cudaMemcpyHostToDevice),
                         "initial cudaMemcpy H2D");
      if (!error.ok()) {
        (void)Release();
        return error;
      }
    }
    for (std::uint32_t index = 0;
         index < std::max<std::uint32_t>(3, buffer_count); ++index) {
      cudaStream_t stream = nullptr;
      error =
          CudaResult(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
      if (!error.ok()) {
        (void)Release();
        return error;
      }
      streams_.push_back(stream);
      cudaEvent_t start = nullptr, stop = nullptr;
      error = CudaResult(cudaEventCreate(&start), "cudaEventCreate(start)");
      if (!error.ok()) {
        (void)Release();
        return error;
      }
      starts_.push_back(start);
      error = CudaResult(cudaEventCreate(&stop), "cudaEventCreate(stop)");
      if (!error.ok()) {
        (void)Release();
        return error;
      }
      stops_.push_back(stop);
    }
    error = CudaResult(cudaEventCreateWithFlags(&gate_, cudaEventDisableTiming),
                       "cudaEventCreate(gate)");
    if (!error.ok()) {
      (void)Release();
      return error;
    }
    gate_release_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!gate_release_) {
      error = {TransferStatus::InternalError,
               static_cast<std::int64_t>(GetLastError()),
               "CreateEventW(copy gate) failed"};
      (void)Release();
      return error;
    }
    prepared_ = true;
    (void)cudaGetLastError();
    return {};
  }

  TransferError Warmup(TransferDirection direction,
                       std::uint64_t bytes) override {
    if (!prepared_ || bytes > capacity_bytes_)
      return {TransferStatus::InternalError, 0,
              "transfer provider is not prepared"};
    const auto kind = direction == TransferDirection::H2D
                          ? cudaMemcpyHostToDevice
                          : cudaMemcpyDeviceToHost;
    void *destination =
        direction == TransferDirection::H2D ? device_[0] : host_[0];
    const void *source =
        direction == TransferDirection::H2D ? host_[0] : device_[0];
    auto result = CudaResult(cudaMemcpyAsync(destination, source,
                                             static_cast<std::size_t>(bytes),
                                             kind, streams_[0]),
                             "warmup cudaMemcpyAsync");
    if (result.ok())
      result = CudaResult(cudaStreamSynchronize(streams_[0]),
                          "warmup cudaStreamSynchronize");
    (void)cudaGetLastError();
    return result;
  }

  TransferSample RunCopy(const CopyRequest &request,
                         std::uint32_t repetition) override {
    TransferSample sample;
    sample.repetition = repetition;
    if (!prepared_ || request.segment_bytes == 0 || request.copy_count == 0) {
      sample.status = TransferStatus::InternalError;
      sample.message = "invalid copy request or unprepared provider";
      return sample;
    }
    const auto required = request.contiguous_segments
                              ? request.segment_bytes * request.copy_count
                              : request.segment_bytes;
    if (required > capacity_bytes_) {
      sample.status = TransferStatus::InternalError;
      sample.message = "copy request exceeds prepared capacity";
      return sample;
    }
    sample.payload_bytes = request.segment_bytes * request.copy_count;
    const auto kind = request.direction == TransferDirection::H2D
                          ? cudaMemcpyHostToDevice
                          : cudaMemcpyDeviceToHost;
    auto error =
        CudaResult(cudaEventRecord(starts_[0],
                                   request.api_mode == CopyApiMode::Asynchronous
                                       ? streams_[0]
                                       : nullptr),
                   "cudaEventRecord(start)");
    const auto end_start = NowNs();
    std::uint64_t api_time = 0;
    for (std::uint32_t copy = 0; copy < request.copy_count && error.ok();
         ++copy) {
      const auto offset =
          request.contiguous_segments ? request.segment_bytes * copy : 0;
      auto *host = static_cast<std::uint8_t *>(host_[0]) + offset;
      auto *device = static_cast<std::uint8_t *>(device_[0]) + offset;
      void *destination = request.direction == TransferDirection::H2D
                              ? static_cast<void *>(device)
                              : static_cast<void *>(host);
      const void *source = request.direction == TransferDirection::H2D
                               ? static_cast<const void *>(host)
                               : static_cast<const void *>(device);
      const auto api_start = NowNs();
      if (request.api_mode == CopyApiMode::Synchronous)
        error = CudaResult(
            cudaMemcpy(destination, source,
                       static_cast<std::size_t>(request.segment_bytes), kind),
            "cudaMemcpy");
      else
        error = CudaResult(
            cudaMemcpyAsync(destination, source,
                            static_cast<std::size_t>(request.segment_bytes),
                            kind, streams_[0]),
            "cudaMemcpyAsync");
      api_time += NowNs() - api_start;
      if (error.ok() && request.synchronize_each &&
          request.api_mode == CopyApiMode::Asynchronous)
        error = CudaResult(cudaStreamSynchronize(streams_[0]),
                           "cudaStreamSynchronize(each)");
    }
    sample.host_api_raw_ns = api_time;
    auto *timing_stream =
        request.api_mode == CopyApiMode::Asynchronous ? streams_[0] : nullptr;
    if (error.ok())
      error = CudaResult(cudaEventRecord(stops_[0], timing_stream),
                         "cudaEventRecord(stop)");
    if (error.ok())
      error = CudaResult(cudaEventSynchronize(stops_[0]),
                         "cudaEventSynchronize(stop)");
    sample.end_to_end_raw_ns = NowNs() - end_start;
    float elapsed_ms = 0;
    if (error.ok())
      error =
          CudaResult(cudaEventElapsedTime(&elapsed_ms, starts_[0], stops_[0]),
                     "cudaEventElapsedTime");
    sample.device_duration_ns = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(elapsed_ms) * 1'000'000.0));
    sample.status = error.status;
    sample.cuda_error = error.native_error;
    sample.message = error.message;
    if (!error.ok())
      (void)cudaGetLastError();
    return sample;
  }

  BidirectionalSample RunBidirectional(std::uint64_t bytes,
                                       std::uint32_t repetition) override {
    BidirectionalSample sample;
    sample.repetition = repetition;
    if (!prepared_ || host_.size() < 2 || bytes > capacity_bytes_) {
      sample.status = TransferStatus::InternalError;
      return sample;
    }
    auto error =
        CudaResult(cudaDeviceSynchronize(), "cudaDeviceSynchronize(gate)");
    if (error.ok())
      error = ResetEvent(gate_release_)
                  ? TransferError{}
                  : TransferError{TransferStatus::InternalError,
                                  static_cast<std::int64_t>(GetLastError()),
                                  "ResetEvent(copy gate) failed"};
    if (error.ok())
      error = CudaResult(cudaLaunchHostFunc(
                             streams_[2],
                             [](void *user_data) {
                               (void)WaitForSingleObject(
                                   static_cast<HANDLE>(user_data), INFINITE);
                             },
                             gate_release_),
                         "cudaLaunchHostFunc(copy gate)");
    if (error.ok())
      error = CudaResult(cudaEventRecord(gate_, streams_[2]),
                         "cudaEventRecord(gate)");
    if (error.ok())
      error = CudaResult(cudaStreamWaitEvent(streams_[0], gate_, 0),
                         "cudaStreamWaitEvent(H2D)");
    if (error.ok())
      error = CudaResult(cudaStreamWaitEvent(streams_[1], gate_, 0),
                         "cudaStreamWaitEvent(D2H)");
    if (error.ok())
      error = CudaResult(cudaEventRecord(starts_[0], streams_[0]),
                         "cudaEventRecord(H2D start)");
    if (error.ok())
      error = CudaResult(cudaEventRecord(starts_[1], streams_[1]),
                         "cudaEventRecord(D2H start)");
    const auto submit_start = NowNs();
    if (error.ok())
      error = CudaResult(cudaMemcpyAsync(device_[0], host_[0],
                                         static_cast<std::size_t>(bytes),
                                         cudaMemcpyHostToDevice, streams_[0]),
                         "bidirectional H2D cudaMemcpyAsync");
    if (error.ok())
      error = CudaResult(cudaMemcpyAsync(host_[1], device_[1],
                                         static_cast<std::size_t>(bytes),
                                         cudaMemcpyDeviceToHost, streams_[1]),
                         "bidirectional D2H cudaMemcpyAsync");
    sample.host_submission_ns = NowNs() - submit_start;
    if (error.ok())
      error = CudaResult(cudaEventRecord(stops_[0], streams_[0]),
                         "cudaEventRecord(H2D stop)");
    if (error.ok())
      error = CudaResult(cudaEventRecord(stops_[1], streams_[1]),
                         "cudaEventRecord(D2H stop)");
    const auto end_start = NowNs();
    if (!SetEvent(gate_release_) && error.ok())
      error = {TransferStatus::InternalError,
               static_cast<std::int64_t>(GetLastError()),
               "SetEvent(copy gate) failed"};
    if (error.ok())
      error = CudaResult(cudaEventSynchronize(stops_[0]),
                         "cudaEventSynchronize(H2D)");
    if (error.ok())
      error = CudaResult(cudaEventSynchronize(stops_[1]),
                         "cudaEventSynchronize(D2H)");
    sample.makespan_ns = NowNs() - end_start;
    float h2d_ms = 0, d2h_ms = 0;
    if (error.ok())
      error = CudaResult(cudaEventElapsedTime(&h2d_ms, starts_[0], stops_[0]),
                         "cudaEventElapsedTime(H2D)");
    if (error.ok())
      error = CudaResult(cudaEventElapsedTime(&d2h_ms, starts_[1], stops_[1]),
                         "cudaEventElapsedTime(D2H)");
    sample.h2d_device_ns = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(h2d_ms) * 1'000'000.0));
    sample.d2h_device_ns = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(d2h_ms) * 1'000'000.0));
    sample.status = error.status;
    return sample;
  }

  TransferError Verify(TransferDirection direction, std::uint64_t bytes,
                       std::uint32_t buffer_index) override {
    if (!prepared_ || buffer_index >= host_.size() || bytes > capacity_bytes_)
      return {TransferStatus::VerificationFailure, 0,
              "invalid verification request"};
    constexpr std::uint32_t samples = 257;
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
      const auto offset = sample == samples - 1
                              ? bytes - 1
                              : (bytes - 1) * sample / (samples - 1);
      std::uint8_t observed = 0;
      if (direction == TransferDirection::H2D) {
        const auto error = CudaResult(
            cudaMemcpy(&observed,
                       static_cast<std::uint8_t *>(device_[buffer_index]) +
                           offset,
                       1, cudaMemcpyDeviceToHost),
            "verification cudaMemcpy D2H");
        if (!error.ok())
          return error;
      } else {
        observed = static_cast<std::uint8_t *>(host_[buffer_index])[offset];
      }
      if (observed != PatternByte(offset, buffer_index))
        return {TransferStatus::VerificationFailure, 0,
                "deterministic transfer pattern mismatch"};
    }
    return {};
  }

  TransferError Release() noexcept override {
    TransferError first;
    const auto remember = [&](TransferError value) {
      if (first.ok() && !value.ok())
        first = std::move(value);
    };
    if (gate_release_)
      (void)SetEvent(gate_release_);
    for (auto event : starts_)
      if (event)
        remember(CudaResult(cudaEventDestroy(event), "cudaEventDestroy(start)",
                            TransferStatus::CleanupFailure));
    for (auto event : stops_)
      if (event)
        remember(CudaResult(cudaEventDestroy(event), "cudaEventDestroy(stop)",
                            TransferStatus::CleanupFailure));
    if (gate_)
      remember(CudaResult(cudaEventDestroy(gate_), "cudaEventDestroy(gate)",
                          TransferStatus::CleanupFailure));
    for (auto stream : streams_)
      if (stream)
        remember(CudaResult(cudaStreamDestroy(stream), "cudaStreamDestroy",
                            TransferStatus::CleanupFailure));
    for (auto pointer : device_)
      if (pointer)
        remember(CudaResult(cudaFree(pointer), "cudaFree",
                            TransferStatus::CleanupFailure));
    for (std::size_t index = 0; index < host_.size(); ++index) {
      bool safe_to_free = true;
      if (index < registered_.size() && registered_[index]) {
        const auto unregister =
            CudaResult(cudaHostUnregister(host_[index]), "cudaHostUnregister",
                       TransferStatus::CleanupFailure);
        remember(unregister);
        safe_to_free = unregister.ok();
      }
      if (!safe_to_free || !host_[index])
        continue;
      if (memory_class_ == HostMemoryClass::PinnedHostAlloc)
        remember(CudaResult(cudaFreeHost(host_[index]), "cudaFreeHost",
                            TransferStatus::CleanupFailure));
      else if (!VirtualFree(host_[index], 0, MEM_RELEASE) && first.ok())
        first = {TransferStatus::CleanupFailure,
                 static_cast<std::int64_t>(GetLastError()),
                 "VirtualFree failed"};
    }
    host_.clear();
    device_.clear();
    registered_.clear();
    streams_.clear();
    starts_.clear();
    stops_.clear();
    gate_ = nullptr;
    if (gate_release_) {
      if (!CloseHandle(gate_release_) && first.ok())
        first = {TransferStatus::CleanupFailure,
                 static_cast<std::int64_t>(GetLastError()),
                 "CloseHandle(copy gate) failed"};
      gate_release_ = nullptr;
    }
    capacity_bytes_ = 0;
    prepared_ = false;
    return first;
  }

private:
  int device_index_{0};
  HostMemoryClass memory_class_{HostMemoryClass::PageablePretouched};
  std::uint64_t capacity_bytes_{0};
  bool prepared_{false};
  std::vector<void *> host_;
  std::vector<void *> device_;
  std::vector<bool> registered_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> starts_;
  std::vector<cudaEvent_t> stops_;
  cudaEvent_t gate_{nullptr};
  HANDLE gate_release_{nullptr};
};

} // namespace

ITransferProvider &NativeTransferProvider() {
  static NativeCudaTransferProvider provider;
  return provider;
}

} // namespace sidecar::cuda
