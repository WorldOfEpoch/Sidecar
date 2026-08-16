#include "sidecar/cuda/overlap.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sidecar::cuda {
namespace {

std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

TransferError Cuda(cudaError_t value, const char *operation,
                   TransferStatus status = TransferStatus::CudaError) {
  if (value == cudaSuccess)
    return {};
  return {value == cudaErrorMemoryAllocation ? TransferStatus::CudaOutOfMemory
                                              : status,
          static_cast<std::int64_t>(value),
          std::string(operation) + ": " + cudaGetErrorString(value)};
}

OverlapStatus StatusFor(const TransferError &error) noexcept {
  return error.ok() ? OverlapStatus::Success : OverlapStatus::CudaError;
}

__global__ void SyntheticAluKernel(float *output,
                                   unsigned long long iterations) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 0.500001f + static_cast<float>(index & 31) * 0.000001f;
  float y = 0.999991f;
  for (unsigned long long i = 0; i < iterations; ++i) {
    x = fmaf(x, y, 0.000013f);
    x = fmaf(x, 0.999983f, -0.000007f);
    x = fmaf(x, y, 0.000011f);
    x = fmaf(x, 0.999979f, -0.000005f);
  }
  output[index] = x;
}

__global__ void MemoryKernel(const float *source, float *destination,
                             std::size_t elements) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto i = index; i < elements; i += stride)
    destination[i] = source[i] + 1.0f;
}

__global__ void FillHalfKernel(__half *value, std::size_t elements,
                               float fill) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < elements)
    value[index] = __float2half(fill);
}

class NativeCudaOverlapProvider final : public IOverlapProvider {
public:
  ~NativeCudaOverlapProvider() override { (void)Release(); }
  bool SupportsCuda() const noexcept override { return true; }

  DeviceMemoryInfo InspectDevice(int device_index) override {
    DeviceMemoryInfo result;
    result.device_index = device_index;
    if (cudaSetDevice(device_index) != cudaSuccess) {
      (void)cudaGetLastError();
      return result;
    }
    cudaDeviceProp properties{};
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      return result;
    }
    result.cuda_supported = true;
    result.name = properties.name;
    result.free_bytes = free_bytes;
    result.total_bytes = total_bytes;
    result.async_engine_count = properties.asyncEngineCount;
    return result;
  }

  TransferTelemetry ReadTelemetry(int device_index) override {
    return NativeTransferProvider().ReadTelemetry(device_index);
  }

  TransferError Prepare(int device_index, HostMemoryClass memory_class,
                        std::uint64_t transfer_capacity_bytes) override {
    auto error = EnsureCore(device_index);
    if (!error.ok())
      return error;
    error = ReleaseTransfer();
    if (!error.ok())
      return error;
    if (transfer_capacity_bytes == 0 ||
        transfer_capacity_bytes > std::numeric_limits<std::size_t>::max())
      return {TransferStatus::HostAllocationFailure, 0,
              "invalid overlap transfer capacity"};
    memory_class_ = memory_class;
    transfer_capacity_bytes_ = transfer_capacity_bytes;
    if (memory_class == HostMemoryClass::PinnedHostAlloc) {
      error = Cuda(cudaHostAlloc(&host_transfer_,
                                 static_cast<std::size_t>(transfer_capacity_bytes),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(overlap)",
                   TransferStatus::HostAllocationFailure);
    } else {
      host_transfer_ = std::malloc(static_cast<std::size_t>(transfer_capacity_bytes));
      if (!host_transfer_)
        error = {TransferStatus::HostAllocationFailure, 0,
                 "malloc(overlap host buffer) failed"};
      if (error.ok() && memory_class == HostMemoryClass::PinnedRegistered) {
        error = Cuda(cudaHostRegister(
                         host_transfer_,
                         static_cast<std::size_t>(transfer_capacity_bytes),
                         cudaHostRegisterDefault),
                     "cudaHostRegister(overlap)",
                     TransferStatus::HostAllocationFailure);
        registered_ = error.ok();
      }
    }
    if (!error.ok()) {
      (void)ReleaseTransfer();
      return error;
    }
    auto *bytes = static_cast<unsigned char *>(host_transfer_);
    for (std::uint64_t i = 0; i < transfer_capacity_bytes; ++i)
      bytes[i] = Pattern(i);
    error = Cuda(cudaMalloc(&device_transfer_,
                            static_cast<std::size_t>(transfer_capacity_bytes)),
                 "cudaMalloc(overlap transfer)");
    if (error.ok())
      error = Cuda(cudaMemcpy(device_transfer_, host_transfer_,
                              static_cast<std::size_t>(transfer_capacity_bytes),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(overlap initialize)");
    if (!error.ok())
      (void)ReleaseTransfer();
    return error;
  }

  WorkloadProfile Calibrate(ComputeWorkload workload, double target_us,
                            std::uint32_t warmups,
                            std::uint32_t repetitions) override {
    WorkloadProfile profile;
    profile.workload = workload;
    profile.target_us = target_us;
    const auto setup = EnsureCore(device_index_);
    if (!setup.ok()) {
      profile.status = OverlapStatus::CudaError;
      profile.message = setup.message;
      return profile;
    }
    profile.memory_working_set_bytes = memory_bytes_;
    profile.memory_block_size = 256;
    profile.memory_elements_per_thread = 0;
    profile.gemm_m = gemm_m_;
    profile.gemm_n = gemm_n_;
    profile.gemm_k = gemm_k_;
    profile.cublas_version = cublas_version_;
    if (workload == ComputeWorkload::SyntheticAlu)
      profile.alu_iterations =
          std::max<std::uint64_t>(1, static_cast<std::uint64_t>(target_us * 64));
    else if (workload == ComputeWorkload::MemoryBound)
      profile.memory_passes = 1;
    else
      profile.gemm_repetitions = 1;

    for (int attempt = 0; attempt < 7; ++attempt) {
      const auto measured = MeasureCompute(profile);
      if (!measured.first.ok()) {
        profile.status = workload == ComputeWorkload::Fp16Gemm
                             ? OverlapStatus::CublasError
                             : OverlapStatus::CudaError;
        profile.message = measured.first.message;
        return profile;
      }
      const double actual_us = measured.second / 1000.0;
      if (actual_us <= 0)
        break;
      const double scale = std::clamp(target_us / actual_us, 0.25, 4.0);
      if (workload == ComputeWorkload::SyntheticAlu)
        profile.alu_iterations = std::max<std::uint64_t>(
            1, static_cast<std::uint64_t>(std::llround(
                   static_cast<double>(profile.alu_iterations) * scale)));
      else if (workload == ComputeWorkload::MemoryBound)
        profile.memory_passes = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::llround(
                   static_cast<double>(profile.memory_passes) * scale)));
      else
        profile.gemm_repetitions = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::llround(
                   static_cast<double>(profile.gemm_repetitions) * scale)));
      if (std::abs(actual_us - target_us) <= std::max(10.0, target_us * 0.03))
        break;
    }
    for (std::uint32_t i = 0; i < warmups; ++i)
      (void)MeasureCompute(profile);
    std::vector<double> values;
    for (std::uint32_t i = 0; i < repetitions; ++i) {
      const auto measured = MeasureCompute(profile);
      if (!measured.first.ok()) {
        profile.status = OverlapStatus::CudaError;
        profile.message = measured.first.message;
        return profile;
      }
      values.push_back(measured.second / 1000.0);
    }
    profile.calibration_samples = static_cast<std::uint32_t>(values.size());
    profile.calibrated_us = trace::CalculateDistribution(std::move(values));
    const auto validation = ValidateCompute(profile);
    profile.validated = validation.ok();
    profile.status = validation.ok() ? OverlapStatus::Success
                                     : OverlapStatus::ComputeValidationFailure;
    profile.message = validation.message;
    return profile;
  }

  RawOverlapTiming RunTopology(const WorkloadProfile &profile,
                               TransferDirection direction,
                               std::uint64_t transfer_bytes,
                               SampleMode sample_mode,
                               InstrumentationMode instrumentation,
                               std::uint64_t gate_delay_ns,
                               std::uint64_t gate_margin_ns,
                               std::uint32_t,
                               std::atomic_bool *external_gate_released) override {
    RawOverlapTiming timing;
    if (!core_ready_ || !host_transfer_ || !device_transfer_ ||
        transfer_bytes == 0 || transfer_bytes > transfer_capacity_bytes_) {
      timing.status = OverlapStatus::InternalError;
      timing.message = "overlap provider is not prepared";
      return timing;
    }
    auto error = Cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(sample)");
    if (error.ok())
      error = ResetComputeBuffers(profile);
    if (!error.ok())
      return Failure(error);
    const bool run_compute = sample_mode != SampleMode::TransferOnly;
    const bool run_transfer = sample_mode != SampleMode::ComputeOnly;
    if (instrumentation == InstrumentationMode::Minimal)
      return RunMinimal(profile, direction, transfer_bytes, run_compute,
                        run_transfer);

    gate_context_.delay_ns = gate_delay_ns;
    gate_context_.external_gate_released = external_gate_released;
    gate_context_.released_ns.store(0, std::memory_order_relaxed);
    gate_queued_ns_ = NowNs();
    error = Cuda(cudaLaunchHostFunc(gate_stream_, HostGateDelay,
                                    &gate_context_),
                 "cudaLaunchHostFunc(gate delay)");
    const bool gate_callback_queued = error.ok();
    if (error.ok())
      error = Cuda(cudaEventRecord(overall_start_, gate_stream_),
                   "cudaEventRecord(overall start)");
    if (error.ok())
      error = Cuda(cudaEventRecord(gate_release_, gate_stream_),
                   "cudaEventRecord(gate release)");
    if (error.ok())
      error = Cuda(cudaStreamWaitEvent(compute_stream_, gate_release_, 0),
                   "cudaStreamWaitEvent(compute)");
    if (error.ok())
      error = Cuda(cudaStreamWaitEvent(transfer_stream_, gate_release_, 0),
                   "cudaStreamWaitEvent(transfer)");
    const auto submission_start = NowNs();
    if (error.ok())
      error = Cuda(cudaEventRecord(compute_start_, compute_stream_),
                   "cudaEventRecord(compute start)");
    if (error.ok() && run_compute)
      error = LaunchCompute(profile, compute_stream_);
    if (error.ok())
      error = Cuda(cudaEventRecord(compute_end_, compute_stream_),
                   "cudaEventRecord(compute end)");
    if (error.ok())
      error = Cuda(cudaEventRecord(transfer_start_, transfer_stream_),
                   "cudaEventRecord(transfer start)");
    if (error.ok() && run_transfer)
      error = LaunchTransfer(direction, transfer_bytes, transfer_stream_);
    if (error.ok())
      error = Cuda(cudaEventRecord(transfer_end_, transfer_stream_),
                   "cudaEventRecord(transfer end)");
    if (error.ok() && instrumentation == InstrumentationMode::Full)
      error = Cuda(cudaStreamWaitEvent(join_stream_, compute_end_, 0),
                   "cudaStreamWaitEvent(join compute)");
    if (error.ok() && instrumentation == InstrumentationMode::Full)
      error = Cuda(cudaStreamWaitEvent(join_stream_, transfer_end_, 0),
                   "cudaStreamWaitEvent(join transfer)");
    if (error.ok() && instrumentation == InstrumentationMode::Full)
      error = Cuda(cudaEventRecord(join_end_, join_stream_),
                   "cudaEventRecord(join end)");
    timing.host_submission_ns = NowNs() - submission_start;
    const auto submission_complete_ns = NowNs();
    if (error.ok())
      error = Cuda(cudaEventSynchronize(overall_start_),
                   "cudaEventSynchronize(overall start)");
    const auto host_start = NowNs();
    if (error.ok()) {
      if (instrumentation == InstrumentationMode::Full)
        error = Cuda(cudaEventSynchronize(join_end_),
                     "cudaEventSynchronize(join end)");
      else
        error = Cuda(cudaEventSynchronize(compute_end_),
                     "cudaEventSynchronize(matched compute end)");
    }
    if (error.ok() && instrumentation != InstrumentationMode::Full)
      error = Cuda(cudaEventSynchronize(transfer_end_),
                   "cudaEventSynchronize(matched transfer end)");
    timing.makespan_host_ns = NowNs() - host_start;
    float compute_ms = 0, transfer_ms = 0, primary_ms = 0;
    float compute_path_ms = 0, transfer_path_ms = 0;
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&compute_ms, compute_start_, compute_end_),
                   "cudaEventElapsedTime(compute)");
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&transfer_ms, transfer_start_,
                                        transfer_end_),
                   "cudaEventElapsedTime(transfer)");
    if (error.ok() && instrumentation == InstrumentationMode::Full)
      error = Cuda(cudaEventElapsedTime(&primary_ms, overall_start_, join_end_),
                   "cudaEventElapsedTime(primary makespan)");
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&compute_path_ms, overall_start_,
                                        compute_end_),
                   "cudaEventElapsedTime(compute crosscheck)");
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&transfer_path_ms, overall_start_,
                                        transfer_end_),
                   "cudaEventElapsedTime(transfer crosscheck)");
    if (!error.ok()) {
      if (gate_callback_queued) (void)cudaStreamSynchronize(gate_stream_);
      gate_context_.external_gate_released = nullptr;
      return Failure(error, timing);
    }
    timing.compute_ns = ToNs(compute_ms);
    timing.transfer_ns = ToNs(transfer_ms);
    const auto released_ns =
        gate_context_.released_ns.load(std::memory_order_acquire);
    timing.gate_actual_ns =
        released_ns > gate_queued_ns_ ? released_ns - gate_queued_ns_ : 0;
    timing.gate_valid = released_ns > submission_complete_ns &&
                        submission_complete_ns + gate_margin_ns < released_ns;
    timing.makespan_device_crosscheck_ns =
        ToNs(std::max(compute_path_ms, transfer_path_ms));
    timing.makespan_device_primary_ns =
        instrumentation == InstrumentationMode::Full
            ? ToNs(primary_ms)
            : timing.makespan_device_crosscheck_ns;
    if (!run_compute)
      timing.compute_ns = 0;
    if (!run_transfer)
      timing.transfer_ns = 0;
    if (!timing.gate_valid)
      timing.status = OverlapStatus::InvalidGate;
    gate_context_.external_gate_released = nullptr;
    return timing;
  }

  TransferError Validate(const WorkloadProfile &profile,
                         TransferDirection direction,
                         std::uint64_t transfer_bytes) override {
    auto error = ValidateCompute(profile);
    if (!error.ok())
      return error;
    if (!host_transfer_ || !device_transfer_ || transfer_bytes == 0 ||
        transfer_bytes > transfer_capacity_bytes_)
      return {TransferStatus::VerificationFailure, 0,
              "invalid overlap transfer validation request"};
    constexpr std::uint32_t kSamples = 257;
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
      const auto offset = sample == kSamples - 1
                              ? transfer_bytes - 1
                              : (transfer_bytes - 1) * sample / (kSamples - 1);
      unsigned char observed = 0;
      if (direction == TransferDirection::H2D) {
        error = Cuda(cudaMemcpy(&observed,
                                static_cast<unsigned char *>(device_transfer_) +
                                    offset,
                                1, cudaMemcpyDeviceToHost),
                     "overlap H2D verification");
      } else {
        observed = static_cast<unsigned char *>(host_transfer_)[offset];
      }
      if (!error.ok())
        return error;
      if (observed != Pattern(offset))
        return {TransferStatus::VerificationFailure, 0,
                "overlap transfer pattern mismatch"};
    }
    return {};
  }

  TransferError Release() noexcept override {
    TransferError first = ReleaseTransfer();
    const auto remember = [&](TransferError value) {
      if (first.ok() && !value.ok())
        first = std::move(value);
    };
    if (cublas_) {
      const auto result = cublasDestroy(cublas_);
      if (result != CUBLAS_STATUS_SUCCESS && first.ok())
        first = {TransferStatus::CleanupFailure,
                 static_cast<std::int64_t>(result), "cublasDestroy failed"};
      cublas_ = nullptr;
    }
    for (auto *pointer : {alu_output_, memory_a_, memory_b_, gemm_a_, gemm_b_,
                          gemm_c_})
      if (pointer)
        remember(Cuda(cudaFree(pointer), "cudaFree(overlap compute)",
                      TransferStatus::CleanupFailure));
    alu_output_ = memory_a_ = memory_b_ = gemm_a_ = gemm_b_ = gemm_c_ =
        nullptr;
    for (auto event : {gate_release_, compute_start_, compute_end_,
                       transfer_start_, transfer_end_, overall_start_,
                       join_end_, minimal_start_, minimal_end_})
      if (event)
        remember(Cuda(cudaEventDestroy(event), "cudaEventDestroy(overlap)",
                      TransferStatus::CleanupFailure));
    gate_release_ = compute_start_ = compute_end_ = transfer_start_ =
        transfer_end_ = overall_start_ = join_end_ = minimal_start_ =
            minimal_end_ = nullptr;
    for (auto stream : {gate_stream_, compute_stream_, transfer_stream_,
                        join_stream_})
      if (stream)
        remember(Cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(overlap)",
                      TransferStatus::CleanupFailure));
    gate_stream_ = compute_stream_ = transfer_stream_ = join_stream_ = nullptr;
    core_ready_ = false;
    return first;
  }

private:
  struct GateContext {
    std::uint64_t delay_ns{0};
    std::atomic<std::uint64_t> released_ns{0};
    std::atomic_bool *external_gate_released{nullptr};
  };
  static void CUDART_CB HostGateDelay(void *opaque) {
    auto *context = static_cast<GateContext *>(opaque);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(context->delay_ns);
    std::this_thread::sleep_until(deadline);
    context->released_ns.store(NowNs(), std::memory_order_release);
    if (context->external_gate_released)
      context->external_gate_released->store(true, std::memory_order_release);
    if (context->external_gate_released)
      context->external_gate_released->notify_all();
  }
  static unsigned char Pattern(std::uint64_t offset) noexcept {
    return static_cast<unsigned char>((offset * 1315423911ULL + 0x6D) & 0xff);
  }
  static std::uint64_t ToNs(float milliseconds) noexcept {
    return static_cast<std::uint64_t>(
        std::llround(static_cast<double>(milliseconds) * 1'000'000.0));
  }
  static RawOverlapTiming Failure(const TransferError &error,
                                  RawOverlapTiming timing = {}) {
    timing.status = StatusFor(error);
    timing.native_error = error.native_error;
    timing.message = error.message;
    return timing;
  }

  TransferError EnsureCore(int device_index) {
    if (core_ready_)
      return {};
    device_index_ = device_index;
    auto error = Cuda(cudaSetDevice(device_index), "cudaSetDevice(overlap)");
    cudaDeviceProp properties{};
    if (error.ok())
      error = Cuda(cudaGetDeviceProperties(&properties, device_index),
                   "cudaGetDeviceProperties(overlap)");
    sm_count_ = std::max(1, properties.multiProcessorCount);
    for (auto **stream : {&gate_stream_, &compute_stream_, &transfer_stream_,
                          &join_stream_})
      if (error.ok())
        error = Cuda(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags(overlap)");
    if (error.ok())
      error = Cuda(cudaEventCreateWithFlags(&gate_release_,
                                            cudaEventDisableTiming),
                   "cudaEventCreate(gate dependency)");
    for (auto **event : {&compute_start_, &compute_end_, &transfer_start_,
                         &transfer_end_, &overall_start_, &join_end_,
                         &minimal_start_, &minimal_end_})
      if (error.ok())
        error = Cuda(cudaEventCreate(event), "cudaEventCreate(overlap timing)");
    alu_elements_ = static_cast<std::size_t>(sm_count_) * 4 * 256;
    memory_bytes_ = 256ULL * 1024 * 1024;
    memory_elements_ = memory_bytes_ / sizeof(float);
    if (error.ok())
      error = Cuda(cudaMalloc(&alu_output_, alu_elements_ * sizeof(float)),
                   "cudaMalloc(ALU output)");
    if (error.ok())
      error = Cuda(cudaMalloc(&memory_a_, memory_bytes_),
                   "cudaMalloc(memory source)");
    if (error.ok())
      error = Cuda(cudaMalloc(&memory_b_, memory_bytes_),
                   "cudaMalloc(memory destination)");
    if (error.ok())
      error = Cuda(cudaMemset(memory_a_, 0, memory_bytes_),
                   "cudaMemset(memory source)");
    if (error.ok())
      error = Cuda(cudaMemset(memory_b_, 0, memory_bytes_),
                   "cudaMemset(memory destination)");
    const auto a_elements = static_cast<std::size_t>(gemm_m_) * gemm_k_;
    const auto b_elements = static_cast<std::size_t>(gemm_k_) * gemm_n_;
    const auto c_elements = static_cast<std::size_t>(gemm_m_) * gemm_n_;
    if (error.ok())
      error = Cuda(cudaMalloc(&gemm_a_, a_elements * sizeof(__half)),
                   "cudaMalloc(GEMM A)");
    if (error.ok())
      error = Cuda(cudaMalloc(&gemm_b_, b_elements * sizeof(__half)),
                   "cudaMalloc(GEMM B)");
    if (error.ok())
      error = Cuda(cudaMalloc(&gemm_c_, c_elements * sizeof(float)),
                   "cudaMalloc(GEMM C)");
    const int block = 256;
    if (error.ok()) {
      FillHalfKernel<<<static_cast<unsigned>((a_elements + block - 1) / block),
                       block, 0, compute_stream_>>>(
          static_cast<__half *>(gemm_a_), a_elements,
          1.0f / std::sqrt(static_cast<float>(gemm_k_)));
      FillHalfKernel<<<static_cast<unsigned>((b_elements + block - 1) / block),
                       block, 0, compute_stream_>>>(
          static_cast<__half *>(gemm_b_), b_elements,
          1.0f / std::sqrt(static_cast<float>(gemm_k_)));
      error = Cuda(cudaGetLastError(), "FillHalfKernel launch");
    }
    if (error.ok()) {
      const auto status = cublasCreate(&cublas_);
      if (status != CUBLAS_STATUS_SUCCESS)
        error = {TransferStatus::CudaError, static_cast<std::int64_t>(status),
                 "cublasCreate failed"};
    }
    if (error.ok()) {
      const auto status = cublasSetStream(cublas_, compute_stream_);
      if (status != CUBLAS_STATUS_SUCCESS)
        error = {TransferStatus::CudaError, static_cast<std::int64_t>(status),
                 "cublasSetStream failed"};
    }
    if (error.ok()) {
      int version = 0;
      const auto status = cublasGetVersion(cublas_, &version);
      if (status == CUBLAS_STATUS_SUCCESS)
        cublas_version_ = version;
    }
    if (error.ok())
      error = Cuda(cudaStreamSynchronize(compute_stream_),
                   "cudaStreamSynchronize(overlap setup)");
    if (!error.ok()) {
      (void)Release();
      return error;
    }
    core_ready_ = true;
    return {};
  }

  TransferError ReleaseTransfer() noexcept {
    TransferError first;
    if (device_transfer_)
      first = Cuda(cudaFree(device_transfer_), "cudaFree(overlap transfer)",
                   TransferStatus::CleanupFailure);
    device_transfer_ = nullptr;
    if (host_transfer_) {
      if (registered_) {
        auto error = Cuda(cudaHostUnregister(host_transfer_),
                          "cudaHostUnregister(overlap)",
                          TransferStatus::CleanupFailure);
        if (first.ok() && !error.ok())
          first = std::move(error);
      }
      if (memory_class_ == HostMemoryClass::PinnedHostAlloc) {
        auto error = Cuda(cudaFreeHost(host_transfer_),
                          "cudaFreeHost(overlap)",
                          TransferStatus::CleanupFailure);
        if (first.ok() && !error.ok())
          first = std::move(error);
      } else {
        std::free(host_transfer_);
      }
    }
    host_transfer_ = nullptr;
    registered_ = false;
    transfer_capacity_bytes_ = 0;
    return first;
  }

  TransferError ResetComputeBuffers(const WorkloadProfile &profile) {
    if (profile.workload != ComputeWorkload::MemoryBound)
      return {};
    auto error = Cuda(cudaMemsetAsync(memory_a_, 0, memory_bytes_,
                                      compute_stream_),
                      "cudaMemsetAsync(memory A reset)");
    if (error.ok())
      error = Cuda(cudaMemsetAsync(memory_b_, 0, memory_bytes_,
                                   compute_stream_),
                   "cudaMemsetAsync(memory B reset)");
    if (error.ok())
      error = Cuda(cudaStreamSynchronize(compute_stream_),
                   "cudaStreamSynchronize(memory reset)");
    return error;
  }

  TransferError LaunchCompute(const WorkloadProfile &profile,
                              cudaStream_t stream) {
    if (profile.workload == ComputeWorkload::SyntheticAlu) {
      SyntheticAluKernel<<<sm_count_ * 4, 256, 0, stream>>>(
          static_cast<float *>(alu_output_), profile.alu_iterations);
      return Cuda(cudaGetLastError(), "SyntheticAluKernel launch");
    }
    if (profile.workload == ComputeWorkload::MemoryBound) {
      auto *source = static_cast<float *>(memory_a_);
      auto *destination = static_cast<float *>(memory_b_);
      for (std::uint32_t pass = 0; pass < profile.memory_passes; ++pass) {
        MemoryKernel<<<sm_count_ * 8, 256, 0, stream>>>(source, destination,
                                                        memory_elements_);
        std::swap(source, destination);
      }
      return Cuda(cudaGetLastError(), "MemoryKernel launch");
    }
    const float alpha = 1.0f, beta = 0.0f;
    for (std::uint32_t repetition = 0;
         repetition < profile.gemm_repetitions; ++repetition) {
      const auto status = cublasGemmEx(
          cublas_, CUBLAS_OP_N, CUBLAS_OP_N, gemm_m_, gemm_n_, gemm_k_,
          &alpha, gemm_a_, CUDA_R_16F, gemm_m_, gemm_b_, CUDA_R_16F, gemm_k_,
          &beta, gemm_c_, CUDA_R_32F, gemm_m_, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT);
      if (status != CUBLAS_STATUS_SUCCESS)
        return {TransferStatus::CudaError, static_cast<std::int64_t>(status),
                "cublasGemmEx failed"};
    }
    return {};
  }

  TransferError LaunchTransfer(TransferDirection direction, std::uint64_t bytes,
                               cudaStream_t stream) {
    const auto kind = direction == TransferDirection::H2D
                          ? cudaMemcpyHostToDevice
                          : cudaMemcpyDeviceToHost;
    void *destination = direction == TransferDirection::H2D ? device_transfer_
                                                             : host_transfer_;
    const void *source = direction == TransferDirection::H2D ? host_transfer_
                                                              : device_transfer_;
    return Cuda(cudaMemcpyAsync(destination, source,
                                static_cast<std::size_t>(bytes), kind, stream),
                "cudaMemcpyAsync(overlap)");
  }

  std::pair<TransferError, std::uint64_t>
  MeasureCompute(const WorkloadProfile &profile) {
    auto error = ResetComputeBuffers(profile);
    if (error.ok())
      error = Cuda(cudaEventRecord(minimal_start_, compute_stream_),
                   "cudaEventRecord(calibration start)");
    if (error.ok())
      error = LaunchCompute(profile, compute_stream_);
    if (error.ok())
      error = Cuda(cudaEventRecord(minimal_end_, compute_stream_),
                   "cudaEventRecord(calibration end)");
    if (error.ok())
      error = Cuda(cudaEventSynchronize(minimal_end_),
                   "cudaEventSynchronize(calibration)");
    float elapsed = 0;
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&elapsed, minimal_start_, minimal_end_),
                   "cudaEventElapsedTime(calibration)");
    return {error, ToNs(elapsed)};
  }

  RawOverlapTiming RunMinimal(const WorkloadProfile &profile,
                              TransferDirection direction,
                              std::uint64_t transfer_bytes, bool run_compute,
                              bool run_transfer) {
    RawOverlapTiming timing;
    auto stream = run_compute ? compute_stream_ : transfer_stream_;
    auto error = Cuda(cudaEventRecord(minimal_start_, stream),
                      "cudaEventRecord(minimal start)");
    const auto host_start = NowNs();
    if (error.ok() && run_compute)
      error = LaunchCompute(profile, stream);
    if (error.ok() && run_transfer)
      error = LaunchTransfer(direction, transfer_bytes, stream);
    timing.host_submission_ns = NowNs() - host_start;
    if (error.ok())
      error = Cuda(cudaEventRecord(minimal_end_, stream),
                   "cudaEventRecord(minimal end)");
    if (error.ok())
      error = Cuda(cudaEventSynchronize(minimal_end_),
                   "cudaEventSynchronize(minimal end)");
    timing.makespan_host_ns = NowNs() - host_start;
    float elapsed = 0;
    if (error.ok())
      error = Cuda(cudaEventElapsedTime(&elapsed, minimal_start_, minimal_end_),
                   "cudaEventElapsedTime(minimal)");
    if (!error.ok())
      return Failure(error, timing);
    const auto ns = ToNs(elapsed);
    timing.compute_ns = run_compute ? ns : 0;
    timing.transfer_ns = run_transfer ? ns : 0;
    timing.makespan_device_primary_ns = ns;
    timing.makespan_device_crosscheck_ns = ns;
    timing.gate_valid = true;
    return timing;
  }

  TransferError ValidateCompute(const WorkloadProfile &profile) {
    auto measured = MeasureCompute(profile);
    if (!measured.first.ok())
      return measured.first;
    if (profile.workload == ComputeWorkload::SyntheticAlu) {
      float first = 0, second = 0;
      auto error = Cuda(cudaMemcpy(&first, alu_output_, sizeof(first),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(ALU validation first)");
      if (error.ok())
        error = MeasureCompute(profile).first;
      if (error.ok())
        error = Cuda(cudaMemcpy(&second, alu_output_, sizeof(second),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy(ALU validation second)");
      if (!error.ok())
        return error;
      if (std::memcmp(&first, &second, sizeof(first)) != 0 ||
          !std::isfinite(first))
        return {TransferStatus::VerificationFailure, 0,
                "SYNTHETIC_ALU deterministic validation failed"};
    } else if (profile.workload == ComputeWorkload::MemoryBound) {
      float observed = 0;
      auto *final = profile.memory_passes % 2 ? memory_b_ : memory_a_;
      auto error = Cuda(cudaMemcpy(&observed, final, sizeof(observed),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(memory validation)");
      if (!error.ok())
        return error;
      if (std::abs(observed - static_cast<float>(profile.memory_passes)) >
          0.01f)
        return {TransferStatus::VerificationFailure, 0,
                "MEMORY_BOUND deterministic validation failed"};
    } else {
      float observed = 0;
      auto error = Cuda(cudaMemcpy(&observed, gemm_c_, sizeof(observed),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(GEMM validation)");
      if (!error.ok())
        return error;
      if (!std::isfinite(observed) || std::abs(observed - 1.0f) > 0.05f)
        return {TransferStatus::VerificationFailure, 0,
                "FP16_GEMM deterministic validation failed"};
    }
    return {};
  }

  int device_index_{0}, sm_count_{1};
  GateContext gate_context_;
  std::uint64_t gate_queued_ns_{0};
  bool core_ready_{false}, registered_{false};
  HostMemoryClass memory_class_{HostMemoryClass::PinnedHostAlloc};
  std::uint64_t transfer_capacity_bytes_{0};
  void *host_transfer_{nullptr}, *device_transfer_{nullptr};
  void *alu_output_{nullptr}, *memory_a_{nullptr}, *memory_b_{nullptr};
  void *gemm_a_{nullptr}, *gemm_b_{nullptr}, *gemm_c_{nullptr};
  std::size_t alu_elements_{0}, memory_bytes_{0}, memory_elements_{0};
  int gemm_m_{2048}, gemm_n_{2048}, gemm_k_{2048};
  std::optional<std::int64_t> cublas_version_;
  cublasHandle_t cublas_{nullptr};
  cudaStream_t gate_stream_{nullptr}, compute_stream_{nullptr};
  cudaStream_t transfer_stream_{nullptr}, join_stream_{nullptr};
  cudaEvent_t gate_release_{nullptr}, compute_start_{nullptr};
  cudaEvent_t compute_end_{nullptr}, transfer_start_{nullptr};
  cudaEvent_t transfer_end_{nullptr}, overall_start_{nullptr};
  cudaEvent_t join_end_{nullptr}, minimal_start_{nullptr};
  cudaEvent_t minimal_end_{nullptr};
};

} // namespace

IOverlapProvider &NativeOverlapProvider() {
  static NativeCudaOverlapProvider provider;
  return provider;
}

} // namespace sidecar::cuda
