#include "sidecar/cuda/transfer.hpp"

#include "sidecar/core/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>

namespace sidecar::cuda {
namespace {

constexpr std::uint64_t kKiB = 1024ULL;
constexpr std::uint64_t kMiB = 1024ULL * 1024;
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;

std::string Escape(const std::string &value) {
  std::ostringstream out;
  for (const unsigned char character : value) {
    if (character == '"')
      out << "\\\"";
    else if (character == '\\')
      out << "\\\\";
    else if (character == '\n')
      out << "\\n";
    else if (character < 0x20)
      out << '?';
    else
      out << static_cast<char>(character);
  }
  return out.str();
}

std::string FormatBytes(std::uint64_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  if (bytes < kMiB)
    out << static_cast<double>(bytes) / kKiB << " KiB";
  else if (bytes < kGiB)
    out << static_cast<double>(bytes) / kMiB << " MiB";
  else
    out << static_cast<double>(bytes) / kGiB << " GiB";
  return out.str();
}

double Median(const std::vector<std::uint64_t> &values) {
  std::vector<double> converted;
  converted.reserve(values.size());
  for (const auto value : values)
    converted.push_back(static_cast<double>(value));
  return trace::CalculateDistribution(std::move(converted)).median;
}

std::string ClassifyAsync(const TransferStatistics &statistics) {
  if (!statistics.count || statistics.end_to_end_ns.median <= 0)
    return "INCONCLUSIVE";
  const auto ratio =
      statistics.host_api_ns.median / statistics.end_to_end_ns.median;
  if (ratio <= 0.25)
    return "HOST_RETURNED_QUICKLY";
  if (ratio >= 0.80)
    return "EFFECTIVELY_SYNCHRONOUS";
  if (ratio >= 0.50)
    return "HOST_STAGING_BLOCK_OBSERVED";
  return "INCONCLUSIVE";
}

std::vector<HostMemoryClass> MemoryOrder(std::vector<HostMemoryClass> values,
                                         std::uint64_t seed) {
  if (!values.empty()) {
    const auto rotation = static_cast<std::size_t>(seed % values.size());
    std::rotate(values.begin(), values.begin() + rotation, values.end());
  }
  return values;
}

memory::MemoryMethod HostMethod(HostMemoryClass memory_class) {
  switch (memory_class) {
  case HostMemoryClass::PageablePretouched:
    return memory::MemoryMethod::Pageable;
  case HostMemoryClass::PinnedHostAlloc:
    return memory::MemoryMethod::CudaHostAlloc;
  case HostMemoryClass::PinnedRegistered:
    return memory::MemoryMethod::CudaHostRegister;
  }
  return memory::MemoryMethod::Pageable;
}

void TelemetryMaximum(TransferTelemetry &target,
                      const TransferTelemetry &observation) {
  target.available = target.available || observation.available;
  const auto maximum = [](auto &destination, const auto &source) {
    if (source && (!destination || *source > *destination))
      destination = source;
  };
  maximum(target.temperature_c, observation.temperature_c);
  maximum(target.graphics_clock_mhz, observation.graphics_clock_mhz);
  maximum(target.memory_clock_mhz, observation.memory_clock_mhz);
  maximum(target.power_watts, observation.power_watts);
  maximum(target.power_limit_watts, observation.power_limit_watts);
  maximum(target.pcie_generation, observation.pcie_generation);
  maximum(target.pcie_width, observation.pcie_width);
}

} // namespace

std::vector<std::uint64_t> DefaultTransferSizeSweep() {
  return {4 * kKiB,   16 * kKiB,  64 * kKiB,  256 * kKiB, 1 * kMiB,   2 * kMiB,
          4 * kMiB,   8 * kMiB,   16 * kMiB,  32 * kMiB,  64 * kMiB,  96 * kMiB,
          128 * kMiB, 192 * kMiB, 256 * kMiB, 384 * kMiB, 512 * kMiB, 1 * kGiB};
}

std::uint32_t AdaptiveRepetitions(std::uint64_t bytes) noexcept {
  if (bytes <= 64 * kKiB)
    return 1000;
  if (bytes <= 256 * kKiB)
    return 500;
  if (bytes <= 1 * kMiB)
    return 300;
  if (bytes <= 2 * kMiB)
    return 200;
  if (bytes <= 4 * kMiB)
    return 150;
  if (bytes <= 8 * kMiB)
    return 100;
  if (bytes <= 16 * kMiB)
    return 80;
  if (bytes <= 32 * kMiB)
    return 60;
  if (bytes <= 64 * kMiB)
    return 40;
  if (bytes <= 96 * kMiB)
    return 30;
  if (bytes <= 128 * kMiB)
    return 25;
  if (bytes <= 192 * kMiB)
    return 20;
  if (bytes <= 256 * kMiB)
    return 15;
  if (bytes <= 384 * kMiB)
    return 10;
  if (bytes <= 512 * kMiB)
    return 8;
  return 5;
}

double BytesPerSecond(std::uint64_t bytes, std::uint64_t duration_ns) noexcept {
  if (duration_ns == 0)
    return 0.0;
  return static_cast<double>(bytes) * 1'000'000'000.0 /
         static_cast<double>(duration_ns);
}

double DecimalGigabytesPerSecond(double bytes_per_second) noexcept {
  return bytes_per_second / 1'000'000'000.0;
}

double BinaryGibibytesPerSecond(double bytes_per_second) noexcept {
  return bytes_per_second / static_cast<double>(kGiB);
}

DeviceSafetyDecision PlanDeviceMemory(const DeviceMemoryInfo &device,
                                      std::uint64_t required_bytes,
                                      const DeviceSafetyPolicy &policy) {
  DeviceSafetyDecision decision;
  decision.required_bytes = required_bytes;
  decision.reserve_bytes = std::max(
      policy.absolute_reserve_bytes,
      static_cast<std::uint64_t>(static_cast<long double>(device.total_bytes) *
                                 policy.total_reserve_fraction));
  if (!device.cuda_supported || device.total_bytes == 0 ||
      device.free_bytes == 0) {
    decision.reason = "CUDA device-memory state is unavailable";
    return decision;
  }
  if (device.free_bytes <= decision.reserve_bytes ||
      required_bytes > device.free_bytes - decision.reserve_bytes) {
    decision.reason = "device allocation would violate the VRAM reserve";
    return decision;
  }
  decision.safe = true;
  decision.reason = "device-memory headroom permits the allocation";
  return decision;
}

std::string
TransferConfigurationIdentity(const TransferConfiguration &configuration) {
  std::ostringstream canonical;
  canonical << "SIDECAR-CUDA-TRANSFER-CONFIG-V1\n"
            << "direction=" << ToString(configuration.direction) << '\n'
            << "memory_class=" << ToString(configuration.memory_class) << '\n'
            << "api_mode=" << ToString(configuration.api_mode) << '\n'
            << "transfer_bytes=" << configuration.transfer_bytes << '\n'
            << "stream_mode=" << configuration.stream_mode << '\n'
            << "batch_count=" << configuration.batch_count << '\n'
            << "chunk_count=" << configuration.chunk_count << '\n'
            << "warmup_count=" << configuration.warmup_count << '\n'
            << "repetitions=" << configuration.repetitions << '\n'
            << "device_index=" << configuration.device_index << '\n'
            << "validation_mode=" << configuration.validation_mode << '\n'
            << "experiment=" << configuration.experiment << '\n';
  return core::Sha256Hex(canonical.str());
}

TransferStatistics
CalculateTransferStatistics(const std::vector<TransferSample> &samples) {
  TransferStatistics statistics;
  std::vector<double> host, device, end_to_end, bandwidth;
  for (const auto &sample : samples) {
    if (sample.status != TransferStatus::Success)
      continue;
    host.push_back(static_cast<double>(sample.host_api_raw_ns));
    device.push_back(static_cast<double>(sample.device_duration_ns));
    end_to_end.push_back(static_cast<double>(sample.end_to_end_raw_ns));
    bandwidth.push_back(
        BytesPerSecond(sample.payload_bytes, sample.device_duration_ns));
  }
  statistics.count = device.size();
  statistics.host_api_ns = trace::CalculateDistribution(std::move(host));
  statistics.device_ns = trace::CalculateDistribution(std::move(device));
  statistics.end_to_end_ns =
      trace::CalculateDistribution(std::move(end_to_end));
  statistics.bytes_per_second =
      trace::CalculateDistribution(std::move(bandwidth));
  statistics.p99_meaningful = statistics.count >= 100;
  return statistics;
}

SaturationProfile
CalculateSaturationProfile(TransferDirection direction,
                           HostMemoryClass memory_class, CopyApiMode api_mode,
                           const std::vector<TransferResult> &results) {
  SaturationProfile profile{direction, memory_class, api_mode};
  std::vector<std::pair<std::uint64_t, double>> curve;
  for (const auto &result : results) {
    if (result.status != TransferStatus::Success ||
        result.configuration.direction != direction ||
        result.configuration.memory_class != memory_class ||
        result.configuration.api_mode != api_mode || !result.statistics.count)
      continue;
    const auto bandwidth = result.statistics.bytes_per_second.median;
    curve.emplace_back(result.configuration.transfer_bytes, bandwidth);
    if (bandwidth > profile.peak_bytes_per_second) {
      profile.peak_bytes_per_second = bandwidth;
      profile.peak_size_bytes = result.configuration.transfer_bytes;
    }
  }
  std::sort(curve.begin(), curve.end());
  const auto knee = [&](double ratio) {
    for (const auto &[bytes, bandwidth] : curve)
      if (bandwidth >= ratio * profile.peak_bytes_per_second)
        return bytes;
    return std::uint64_t{0};
  };
  profile.knee_80_bytes = knee(0.80);
  profile.knee_90_bytes = knee(0.90);
  profile.knee_95_bytes = knee(0.95);
  return profile;
}

double CalculateConcurrencyBenefit(double isolated_h2d_ns,
                                   double isolated_d2h_ns,
                                   double concurrent_makespan_ns) noexcept {
  if (concurrent_makespan_ns <= 0)
    return 0.0;
  return (isolated_h2d_ns + isolated_d2h_ns) / concurrent_makespan_ns;
}

std::uint64_t ChunkSize(std::uint64_t total_bytes, std::uint32_t chunk_count) {
  if (chunk_count == 0 || total_bytes == 0 || total_bytes % chunk_count != 0)
    throw std::invalid_argument(
        "total payload must divide evenly into non-zero chunks");
  return total_bytes / chunk_count;
}

std::uint64_t BatchPayloadBytes(std::uint64_t copy_bytes,
                                std::uint32_t batch_count) {
  if (copy_bytes == 0 || batch_count == 0 ||
      copy_bytes > UINT64_MAX / batch_count)
    throw std::invalid_argument("invalid batch payload");
  return copy_bytes * batch_count;
}

TransferRunReport RunTransferSweep(ITransferProvider &provider,
                                   const memory::MemorySnapshot &host,
                                   const TransferRunOptions &options) {
  TransferRunReport report;
  report.host_timer = memory::CalibrateTimer();
  report.device = provider.InspectDevice(options.device_index);
  report.ordering_seed = options.ordering_seed;
  if (!provider.SupportsCuda() || !report.device.cuda_supported) {
    report.status = TransferStatus::SkippedUnsupported;
    report.message = "CUDA transfer support is unavailable";
    return report;
  }
  const auto sizes =
      options.sizes.empty() ? DefaultTransferSizeSweep() : options.sizes;
  const auto memories = MemoryOrder(
      options.memory_classes.empty()
          ? std::vector<HostMemoryClass>{HostMemoryClass::PageablePretouched,
                                         HostMemoryClass::PinnedHostAlloc,
                                         HostMemoryClass::PinnedRegistered}
          : options.memory_classes,
      options.ordering_seed);
  const auto directions =
      options.directions.empty()
          ? std::vector<TransferDirection>{TransferDirection::H2D,
                                           TransferDirection::D2H}
          : options.directions;
  const auto apis = options.api_modes.empty()
                        ? std::vector<CopyApiMode>{CopyApiMode::Synchronous,
                                                   CopyApiMode::Asynchronous}
                        : options.api_modes;
  const auto capacity = *std::max_element(sizes.begin(), sizes.end());
  const auto device_safety =
      PlanDeviceMemory(report.device, capacity, options.device_safety);
  if (!device_safety.safe) {
    report.status = TransferStatus::SkippedSafetyLimit;
    report.message = device_safety.reason;
    return report;
  }

  for (const auto memory_class : memories) {
    const auto host_method = HostMethod(memory_class);
    const auto host_decision = memory::PlanAllocation(host, options.host_safety,
                                                      host_method, capacity);
    if (!host_decision.safe()) {
      report.status = TransferStatus::SkippedSafetyLimit;
      report.message = host_decision.reason;
      continue;
    }
    if (options.dry_run)
      continue;
    const auto prepared =
        provider.Prepare(options.device_index, memory_class, capacity, 1);
    if (!prepared.ok()) {
      report.status = prepared.status;
      report.message = prepared.message;
      break;
    }
    for (std::size_t api_index = 0; api_index < apis.size(); ++api_index) {
      auto ordered_sizes = sizes;
      if (api_index % 2 == 1)
        std::reverse(ordered_sizes.begin(), ordered_sizes.end());
      for (std::size_t size_index = 0; size_index < ordered_sizes.size();
           ++size_index) {
        auto ordered_directions = directions;
        if (size_index % 2 == 1)
          std::reverse(ordered_directions.begin(), ordered_directions.end());
        for (const auto direction : ordered_directions) {
          TransferResult result;
          result.configuration.direction = direction;
          result.configuration.memory_class = memory_class;
          result.configuration.api_mode = apis[api_index];
          result.configuration.transfer_bytes = ordered_sizes[size_index];
          result.configuration.warmup_count = options.warmup_count;
          result.configuration.repetitions = options.repetitions.value_or(
              AdaptiveRepetitions(ordered_sizes[size_index]));
          result.configuration.device_index = options.device_index;
          TransferError warmup;
          for (std::uint32_t warm = 0; warm < options.warmup_count; ++warm) {
            warmup = provider.Warmup(direction, ordered_sizes[size_index]);
            if (!warmup.ok())
              break;
          }
          if (!warmup.ok()) {
            result.status = warmup.status;
            result.message = warmup.message;
            report.status = warmup.status;
            report.message = warmup.message;
            report.results.push_back(std::move(result));
            continue;
          }
          CopyRequest request{
              direction, apis[api_index], ordered_sizes[size_index],
              1,         false,           false};
          for (std::uint32_t repetition = 0;
               repetition < result.configuration.repetitions; ++repetition) {
            auto sample = provider.RunCopy(request, repetition);
            const auto status = sample.status;
            result.samples.push_back(std::move(sample));
            if (status != TransferStatus::Success)
              break;
          }
          const auto verification =
              provider.Verify(direction, ordered_sizes[size_index]);
          result.verified = verification.ok();
          result.statistics = CalculateTransferStatistics(result.samples);
          result.status =
              verification.ok() ? TransferStatus::Success : verification.status;
          result.message = verification.message;
          if (result.configuration.api_mode == CopyApiMode::Asynchronous)
            result.async_behavior = ClassifyAsync(result.statistics);
          for (const auto &sample : result.samples) {
            if (sample.status != TransferStatus::Success) {
              result.status = sample.status;
              result.message = sample.message;
              break;
            }
          }
          if (result.status != TransferStatus::Success) {
            report.status = result.status;
            report.message = result.message;
          }
          report.results.push_back(std::move(result));
        }
      }
    }
    const auto released = provider.Release();
    if (!released.ok()) {
      report.status = released.status;
      report.message = released.message;
      break;
    }
  }

  for (const auto memory_class : memories)
    for (const auto direction : directions)
      for (const auto api : apis)
        report.profiles.push_back(CalculateSaturationProfile(
            direction, memory_class, api, report.results));

  std::set<std::uint64_t> candidates;
  const auto latency_candidate =
      std::min_element(sizes.begin(), sizes.end(), [](auto left, auto right) {
        const auto left_distance = left >= kMiB ? left - kMiB : UINT64_MAX;
        const auto right_distance = right >= kMiB ? right - kMiB : UINT64_MAX;
        return left_distance < right_distance;
      });
  if (latency_candidate != sizes.end() && *latency_candidate >= kMiB)
    candidates.insert(*latency_candidate);
  if (std::find(sizes.begin(), sizes.end(), 96 * kMiB) != sizes.end())
    candidates.insert(96 * kMiB);
  for (const auto &profile : report.profiles) {
    if (profile.api_mode != CopyApiMode::Asynchronous ||
        profile.memory_class == HostMemoryClass::PageablePretouched)
      continue;
    if (profile.knee_95_bytes)
      candidates.insert(profile.knee_95_bytes);
    if (profile.peak_size_bytes)
      candidates.insert(profile.peak_size_bytes);
  }
  candidates.insert(*std::max_element(sizes.begin(), sizes.end()));
  report.wu6_candidate_sizes.assign(candidates.begin(), candidates.end());
  if (report.wu6_candidate_sizes.size() > 6) {
    const auto largest = report.wu6_candidate_sizes.back();
    report.wu6_candidate_sizes.resize(5);
    report.wu6_candidate_sizes.push_back(largest);
  }
  return report;
}

std::vector<BidirectionalResult> RunBidirectionalSweep(
    ITransferProvider &provider, const DeviceMemoryInfo &device,
    const memory::MemorySnapshot &host, const std::vector<std::uint64_t> &sizes,
    HostMemoryClass memory_class, std::uint32_t repetitions,
    const memory::SafetyPolicy &host_safety,
    const DeviceSafetyPolicy &device_safety) {
  std::vector<BidirectionalResult> results;
  if (sizes.empty() || !provider.SupportsCuda())
    return results;
  const auto capacity = *std::max_element(sizes.begin(), sizes.end());
  const auto host_method = HostMethod(memory_class);
  if (!memory::PlanAllocation(host, host_safety, host_method, capacity * 2)
           .safe() ||
      !PlanDeviceMemory(device, capacity * 2, device_safety).safe)
    return results;
  const auto prepared =
      provider.Prepare(device.device_index, memory_class, capacity, 2);
  if (!prepared.ok())
    return results;
  for (const auto bytes : sizes) {
    BidirectionalResult result;
    result.memory_class = memory_class;
    result.transfer_bytes = bytes;
    result.repetitions = repetitions;
    auto warmup = provider.Warmup(TransferDirection::H2D, bytes);
    if (warmup.ok())
      warmup = provider.Warmup(TransferDirection::D2H, bytes);
    if (!warmup.ok()) {
      result.status = warmup.status;
      result.message = warmup.message;
      results.push_back(std::move(result));
      continue;
    }
    std::vector<std::uint64_t> h0, d0, hc, dc, makespans;
    for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition) {
      auto h =
          provider.RunCopy({TransferDirection::H2D, CopyApiMode::Asynchronous,
                            bytes, 1, false, false},
                           repetition);
      auto d =
          provider.RunCopy({TransferDirection::D2H, CopyApiMode::Asynchronous,
                            bytes, 1, false, false},
                           repetition);
      auto concurrent = provider.RunBidirectional(bytes, repetition);
      if (h.status != TransferStatus::Success ||
          d.status != TransferStatus::Success ||
          concurrent.status != TransferStatus::Success) {
        result.status =
            h.status != TransferStatus::Success
                ? h.status
                : (d.status != TransferStatus::Success ? d.status
                                                       : concurrent.status);
        break;
      }
      h0.push_back(h.device_duration_ns);
      d0.push_back(d.device_duration_ns);
      hc.push_back(concurrent.h2d_device_ns);
      dc.push_back(concurrent.d2h_device_ns);
      makespans.push_back(concurrent.makespan_ns);
      result.samples.push_back(concurrent);
    }
    result.isolated_h2d_median_ns = Median(h0);
    result.isolated_d2h_median_ns = Median(d0);
    result.concurrent_h2d_median_ns = Median(hc);
    result.concurrent_d2h_median_ns = Median(dc);
    result.makespan_median_ns = Median(makespans);
    result.aggregate_bytes_per_second = BytesPerSecond(
        bytes * 2, static_cast<std::uint64_t>(result.makespan_median_ns));
    result.concurrency_benefit = CalculateConcurrencyBenefit(
        result.isolated_h2d_median_ns, result.isolated_d2h_median_ns,
        result.makespan_median_ns);
    const auto hverify = provider.Verify(TransferDirection::H2D, bytes, 0);
    const auto dverify = provider.Verify(TransferDirection::D2H, bytes, 1);
    result.verified = hverify.ok() && dverify.ok();
    if (!result.verified) {
      result.status = TransferStatus::VerificationFailure;
      result.message = !hverify.ok() ? hverify.message : dverify.message;
    }
    results.push_back(std::move(result));
  }
  const auto released = provider.Release();
  if (!released.ok()) {
    if (results.empty())
      results.push_back({});
    results.back().status = released.status;
    results.back().message = released.message;
  }
  return results;
}

SustainedResult RunSustainedTransfer(ITransferProvider &provider,
                                     const DeviceMemoryInfo &device,
                                     const memory::MemorySnapshot &host,
                                     TransferConfiguration configuration,
                                     std::uint64_t target_payload_bytes,
                                     const memory::SafetyPolicy &host_safety,
                                     const DeviceSafetyPolicy &device_safety) {
  SustainedResult result;
  result.configuration = configuration;
  if (!provider.SupportsCuda() || !device.cuda_supported) {
    result.status = TransferStatus::SkippedUnsupported;
    result.message = "CUDA transfer support is unavailable";
    return result;
  }
  const auto host_method = HostMethod(configuration.memory_class);
  if (!memory::PlanAllocation(host, host_safety, host_method,
                              configuration.transfer_bytes)
           .safe() ||
      !PlanDeviceMemory(device, configuration.transfer_bytes, device_safety)
           .safe) {
    result.status = TransferStatus::SkippedSafetyLimit;
    result.message =
        "host or device safety reserve rejected sustained configuration";
    return result;
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));
  result.before = provider.ReadTelemetry(configuration.device_index);
  const auto prepared =
      provider.Prepare(configuration.device_index, configuration.memory_class,
                       configuration.transfer_bytes, 1);
  if (!prepared.ok()) {
    result.status = prepared.status;
    result.message = prepared.message;
    return result;
  }
  const auto warmup =
      provider.Warmup(configuration.direction, configuration.transfer_bytes);
  if (!warmup.ok()) {
    result.status = warmup.status;
    result.message = warmup.message;
    const auto released = provider.Release();
    if (!released.ok()) {
      result.status = released.status;
      result.message = released.message;
    }
    return result;
  }
  const auto copies = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
      (target_payload_bytes + configuration.transfer_bytes - 1) /
          configuration.transfer_bytes,
      2, 4096));
  result.configuration.batch_count = copies;
  result.configuration.repetitions = 1;
  result.configuration.stream_mode = "SUSTAINED_REUSE_ONE_SYNC";
  std::atomic_bool stop_polling{false};
  std::thread observer([&] {
    while (!stop_polling.load(std::memory_order_relaxed)) {
      TelemetryMaximum(result.maximum_during,
                       provider.ReadTelemetry(configuration.device_index));
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
  auto sample =
      provider.RunCopy({configuration.direction, configuration.api_mode,
                        configuration.transfer_bytes, copies, false, false},
                       0);
  stop_polling.store(true, std::memory_order_relaxed);
  observer.join();
  result.total_payload_bytes = sample.payload_bytes;
  result.total_duration_ns = sample.end_to_end_raw_ns;
  result.host_submission_ns = sample.host_api_raw_ns;
  result.device_duration_ns = sample.device_duration_ns;
  result.bytes_per_second =
      BytesPerSecond(sample.payload_bytes, sample.device_duration_ns);
  result.status = sample.status;
  result.message = sample.message;
  if (sample.status == TransferStatus::Success) {
    const auto verify =
        provider.Verify(configuration.direction, configuration.transfer_bytes);
    result.verified = verify.ok();
    if (!verify.ok()) {
      result.status = verify.status;
      result.message = verify.message;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));
  result.after = provider.ReadTelemetry(configuration.device_index);
  if (result.before.temperature_c && result.after.temperature_c &&
      *result.after.temperature_c > *result.before.temperature_c + 8)
    result.noisy = true;
  const auto released = provider.Release();
  if (!released.ok()) {
    result.status = released.status;
    result.message = released.message;
  }
  return result;
}

std::string TransferRunReportToJson(const TransferRunReport &report) {
  std::ostringstream out;
  out << "{\"status\":\"" << ToString(report.status) << "\",\"message\":\""
      << Escape(report.message)
      << "\",\"device\":{\"index\":" << report.device.device_index
      << ",\"name\":\"" << Escape(report.device.name)
      << "\",\"free_bytes\":" << report.device.free_bytes
      << ",\"total_bytes\":" << report.device.total_bytes
      << ",\"async_engine_count\":" << report.device.async_engine_count
      << "},\"host_timer\":{\"method\":\"" << Escape(report.host_timer.method)
      << "\",\"bracket_overhead_ns\":" << report.host_timer.bracket_overhead_ns
      << ",\"observed_resolution_ns\":"
      << report.host_timer.observed_resolution_ns
      << "},\"ordering_seed\":" << report.ordering_seed << ",\"results\":[";
  for (std::size_t index = 0; index < report.results.size(); ++index) {
    if (index)
      out << ',';
    const auto &result = report.results[index];
    out << "{\"configuration_id\":\""
        << TransferConfigurationIdentity(result.configuration)
        << "\",\"direction\":\"" << ToString(result.configuration.direction)
        << "\",\"memory_class\":\""
        << ToString(result.configuration.memory_class) << "\",\"api_mode\":\""
        << ToString(result.configuration.api_mode)
        << "\",\"transfer_bytes\":" << result.configuration.transfer_bytes
        << ",\"repetitions\":" << result.configuration.repetitions
        << ",\"status\":\"" << ToString(result.status)
        << "\",\"verified\":" << (result.verified ? "true" : "false")
        << ",\"noisy\":" << (result.noisy ? "true" : "false")
        << ",\"async_behavior\":\"" << result.async_behavior
        << "\",\"message\":\"" << Escape(result.message)
        << "\",\"statistics\":{\"count\":" << result.statistics.count
        << ",\"host_api_median_ns\":" << result.statistics.host_api_ns.median
        << ",\"device_median_ns\":" << result.statistics.device_ns.median
        << ",\"end_to_end_median_ns\":"
        << result.statistics.end_to_end_ns.median
        << ",\"median_bytes_per_second\":"
        << result.statistics.bytes_per_second.median << "},\"samples\":[";
    for (std::size_t sample_index = 0; sample_index < result.samples.size();
         ++sample_index) {
      if (sample_index)
        out << ',';
      const auto &sample = result.samples[sample_index];
      out << "{\"repetition\":" << sample.repetition
          << ",\"payload_bytes\":" << sample.payload_bytes
          << ",\"host_api_raw_ns\":" << sample.host_api_raw_ns
          << ",\"device_duration_ns\":" << sample.device_duration_ns
          << ",\"end_to_end_raw_ns\":" << sample.end_to_end_raw_ns
          << ",\"status\":\"" << ToString(sample.status)
          << "\",\"cuda_error\":" << sample.cuda_error << ",\"message\":\""
          << Escape(sample.message) << "\"}";
    }
    out << "]}";
  }
  out << "],\"profiles\":[";
  for (std::size_t index = 0; index < report.profiles.size(); ++index) {
    if (index)
      out << ',';
    const auto &profile = report.profiles[index];
    out << "{\"direction\":\"" << ToString(profile.direction)
        << "\",\"memory_class\":\"" << ToString(profile.memory_class)
        << "\",\"api_mode\":\"" << ToString(profile.api_mode)
        << "\",\"peak_bytes_per_second\":" << profile.peak_bytes_per_second
        << ",\"peak_size_bytes\":" << profile.peak_size_bytes
        << ",\"knee_80_bytes\":" << profile.knee_80_bytes
        << ",\"knee_90_bytes\":" << profile.knee_90_bytes
        << ",\"knee_95_bytes\":" << profile.knee_95_bytes << '}';
  }
  out << "],\"wu6_candidate_sizes\":[";
  for (std::size_t index = 0; index < report.wu6_candidate_sizes.size();
       ++index) {
    if (index)
      out << ',';
    out << report.wu6_candidate_sizes[index];
  }
  out << "]}";
  return out.str();
}

std::string FormatTransferRunReport(const TransferRunReport &report) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << "SIDECAR CUDA TRANSFER PHYSICS\n\nDevice: " << report.device.name
      << "\nAsync engines: " << report.device.async_engine_count
      << "\nHost timer bracket/resolution: "
      << report.host_timer.bracket_overhead_ns << '/'
      << report.host_timer.observed_resolution_ns << " ns"
      << "\nOrdering seed: " << report.ordering_seed
      << "\nStatus: " << ToString(report.status) << "\n";
  if (!report.message.empty())
    out << "Message: " << report.message << '\n';
  for (const auto &result : report.results) {
    out << '\n'
        << ToString(result.configuration.direction) << ' '
        << ToString(result.configuration.memory_class) << ' '
        << ToString(result.configuration.api_mode) << ' '
        << FormatBytes(result.configuration.transfer_bytes) << "  "
        << DecimalGigabytesPerSecond(result.statistics.bytes_per_second.median)
        << " GB/s  "
        << BinaryGibibytesPerSecond(result.statistics.bytes_per_second.median)
        << " GiB/s  device=" << result.statistics.device_ns.median / 1000.0
        << " us device_p95=" << result.statistics.device_ns.p95 / 1000.0
        << " us api=" << result.statistics.host_api_ns.median / 1000.0
        << " us e2e=" << result.statistics.end_to_end_ns.median / 1000.0
        << " us status=" << ToString(result.status);
    if (result.statistics.p99_meaningful)
      out << " device_p99=" << result.statistics.device_ns.p99 / 1000.0
          << " us";
    else
      out << " device_p99=insufficient-samples";
    if (result.configuration.api_mode == CopyApiMode::Asynchronous)
      out << " behavior=" << result.async_behavior;
    out << '\n';
  }
  out << "\nSATURATION PROFILES\n";
  for (const auto &profile : report.profiles) {
    out << ToString(profile.direction) << ' ' << ToString(profile.memory_class)
        << ' ' << ToString(profile.api_mode)
        << " peak=" << DecimalGigabytesPerSecond(profile.peak_bytes_per_second)
        << " GB/s at " << FormatBytes(profile.peak_size_bytes)
        << " knees80/90/95=" << FormatBytes(profile.knee_80_bytes) << '/'
        << FormatBytes(profile.knee_90_bytes) << '/'
        << FormatBytes(profile.knee_95_bytes) << '\n';
  }
  return out.str();
}

std::string
BidirectionalResultsToJson(const std::vector<BidirectionalResult> &results) {
  std::ostringstream out;
  out << "{\"results\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index)
      out << ',';
    const auto &result = results[index];
    out << "{\"memory_class\":\"" << ToString(result.memory_class)
        << "\",\"transfer_bytes\":" << result.transfer_bytes
        << ",\"isolated_h2d_median_ns\":" << result.isolated_h2d_median_ns
        << ",\"isolated_d2h_median_ns\":" << result.isolated_d2h_median_ns
        << ",\"concurrent_h2d_median_ns\":" << result.concurrent_h2d_median_ns
        << ",\"concurrent_d2h_median_ns\":" << result.concurrent_d2h_median_ns
        << ",\"makespan_median_ns\":" << result.makespan_median_ns
        << ",\"h2d_slowdown\":"
        << (result.isolated_h2d_median_ns > 0
                ? result.concurrent_h2d_median_ns /
                      result.isolated_h2d_median_ns
                : 0.0)
        << ",\"d2h_slowdown\":"
        << (result.isolated_d2h_median_ns > 0
                ? result.concurrent_d2h_median_ns /
                      result.isolated_d2h_median_ns
                : 0.0)
        << ",\"aggregate_bytes_per_second\":"
        << result.aggregate_bytes_per_second
        << ",\"concurrency_benefit\":" << result.concurrency_benefit
        << ",\"verified\":" << (result.verified ? "true" : "false")
        << ",\"status\":\"" << ToString(result.status) << "\"}";
  }
  out << "]}";
  return out.str();
}

std::string
FormatBidirectionalResults(const std::vector<BidirectionalResult> &results) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << "SIDECAR CUDA BIDIRECTIONAL COPIES\n";
  for (const auto &result : results)
    out << '\n'
        << ToString(result.memory_class) << ' '
        << FormatBytes(result.transfer_bytes) << " aggregate="
        << DecimalGigabytesPerSecond(result.aggregate_bytes_per_second)
        << " GB/s benefit=" << result.concurrency_benefit << "x h2d_slowdown="
        << (result.isolated_h2d_median_ns > 0
                ? result.concurrent_h2d_median_ns /
                      result.isolated_h2d_median_ns
                : 0.0)
        << "x d2h_slowdown="
        << (result.isolated_d2h_median_ns > 0
                ? result.concurrent_d2h_median_ns /
                      result.isolated_d2h_median_ns
                : 0.0)
        << "x makespan=" << result.makespan_median_ns / 1'000'000.0
        << " ms verified=" << (result.verified ? "yes" : "no")
        << " status=" << ToString(result.status) << '\n';
  return out.str();
}

std::string SustainedResultToJson(const SustainedResult &result) {
  const auto telemetry = [](const TransferTelemetry &value) {
    std::ostringstream out;
    const auto optional = [&](const auto &item) {
      if (item)
        out << *item;
      else
        out << "null";
    };
    out << "{\"temperature_c\":";
    optional(value.temperature_c);
    out << ",\"graphics_clock_mhz\":";
    optional(value.graphics_clock_mhz);
    out << ",\"memory_clock_mhz\":";
    optional(value.memory_clock_mhz);
    out << ",\"power_watts\":";
    optional(value.power_watts);
    out << ",\"power_limit_watts\":";
    optional(value.power_limit_watts);
    out << ",\"pcie_generation\":";
    optional(value.pcie_generation);
    out << ",\"pcie_width\":";
    optional(value.pcie_width);
    out << '}';
    return out.str();
  };
  std::ostringstream out;
  out << "{\"direction\":\"" << ToString(result.configuration.direction)
      << "\",\"memory_class\":\"" << ToString(result.configuration.memory_class)
      << "\",\"transfer_bytes\":" << result.configuration.transfer_bytes
      << ",\"total_payload_bytes\":" << result.total_payload_bytes
      << ",\"total_duration_ns\":" << result.total_duration_ns
      << ",\"host_submission_ns\":" << result.host_submission_ns
      << ",\"device_duration_ns\":" << result.device_duration_ns
      << ",\"bytes_per_second\":" << result.bytes_per_second
      << ",\"verified\":" << (result.verified ? "true" : "false")
      << ",\"noisy\":" << (result.noisy ? "true" : "false") << ",\"status\":\""
      << ToString(result.status) << "\",\"message\":\""
      << Escape(result.message) << "\",\"before\":" << telemetry(result.before)
      << ",\"maximum_during\":" << telemetry(result.maximum_during)
      << ",\"after\":" << telemetry(result.after) << '}';
  return out.str();
}

std::string FormatSustainedResult(const SustainedResult &result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << "SIDECAR CUDA SUSTAINED TRANSFER\n\n"
      << ToString(result.configuration.direction) << ' '
      << ToString(result.configuration.memory_class) << ' '
      << FormatBytes(result.configuration.transfer_bytes) << '\n'
      << "Payload: " << FormatBytes(result.total_payload_bytes) << '\n'
      << "Throughput: " << DecimalGigabytesPerSecond(result.bytes_per_second)
      << " GB/s (" << BinaryGibibytesPerSecond(result.bytes_per_second)
      << " GiB/s)\n"
      << "Active PCIe: Gen" << result.maximum_during.pcie_generation.value_or(0)
      << " x" << result.maximum_during.pcie_width.value_or(0) << '\n'
      << "Temperature before/after: " << result.before.temperature_c.value_or(0)
      << '/' << result.after.temperature_c.value_or(0) << " C\n"
      << "Verified: " << (result.verified ? "yes" : "no")
      << "  Status: " << ToString(result.status) << '\n';
  if (!result.message.empty())
    out << "Message: " << result.message << '\n';
  return out.str();
}

const char *ToString(TransferDirection direction) noexcept {
  return direction == TransferDirection::H2D ? "H2D" : "D2H";
}
const char *ToString(HostMemoryClass memory_class) noexcept {
  switch (memory_class) {
  case HostMemoryClass::PageablePretouched:
    return "PAGEABLE_PRETOUCHED";
  case HostMemoryClass::PinnedHostAlloc:
    return "PINNED_HOSTALLOC";
  case HostMemoryClass::PinnedRegistered:
    return "PINNED_REGISTERED";
  }
  return "PAGEABLE_PRETOUCHED";
}
const char *ToString(CopyApiMode api_mode) noexcept {
  return api_mode == CopyApiMode::Synchronous ? "SYNCHRONOUS" : "ASYNCHRONOUS";
}
const char *ToString(TransferStatus status) noexcept {
  switch (status) {
  case TransferStatus::Success:
    return "SUCCESS";
  case TransferStatus::SkippedSafetyLimit:
    return "SKIPPED_SAFETY_LIMIT";
  case TransferStatus::SkippedUnsupported:
    return "SKIPPED_UNSUPPORTED";
  case TransferStatus::CudaOutOfMemory:
    return "CUDA_OUT_OF_MEMORY";
  case TransferStatus::CudaError:
    return "CUDA_ERROR";
  case TransferStatus::HostAllocationFailure:
    return "HOST_ALLOCATION_FAILURE";
  case TransferStatus::VerificationFailure:
    return "VERIFICATION_FAILURE";
  case TransferStatus::CleanupFailure:
    return "CLEANUP_FAILURE";
  case TransferStatus::InternalError:
    return "INTERNAL_ERROR";
  }
  return "INTERNAL_ERROR";
}

} // namespace sidecar::cuda
