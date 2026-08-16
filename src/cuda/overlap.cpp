#include "sidecar/cuda/overlap.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/memory/safety.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

namespace sidecar::cuda {
namespace {
constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

double Clamp01(double value) noexcept {
  return std::clamp(value, 0.0, 1.0);
}

bool PairedDirectionIsSignificant(const std::vector<double> &differences) {
  std::size_t positive = 0, negative = 0;
  for (const auto difference : differences) {
    positive += difference > 0.0;
    negative += difference < 0.0;
  }
  const auto count = positive + negative;
  const auto same_direction = std::max(positive, negative);
  if (count == 0)
    return false;

  // Exact two-sided sign test. Control suites currently use 31 pairs, but the
  // recurrence remains stable for any practical WU6 control population.
  long double combination = 1.0L;
  long double upper_tail = 0.0L;
  for (std::size_t successes = 0; successes <= count; ++successes) {
    if (successes >= same_direction)
      upper_tail += combination;
    if (successes < count)
      combination *= static_cast<long double>(count - successes) /
                     static_cast<long double>(successes + 1);
  }
  const auto two_sided =
      std::min(1.0L, 2.0L * upper_tail / std::pow(2.0L, int(count)));
  return two_sided < 0.05L;
}

bool ValidForStatistics(const OverlapSample &sample) noexcept {
  return sample.status == OverlapStatus::Success ||
         sample.status == OverlapStatus::MeasurementSensitive;
}

trace::Distribution DistributionOf(const std::vector<OverlapSample> &samples,
                                   double (*select)(const OverlapSample &)) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto &sample : samples)
    if (ValidForStatistics(sample))
      values.push_back(select(sample));
  return trace::CalculateDistribution(std::move(values));
}

std::string Escape(std::string_view value) {
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

std::string DistributionJson(const trace::Distribution &value) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"mean\":" << value.mean
      << ",\"median\":" << value.median << ",\"p50\":" << value.p50
      << ",\"p90\":" << value.p90 << ",\"p95\":" << value.p95
      << ",\"p99\":" << value.p99 << ",\"minimum\":" << value.minimum
      << ",\"maximum\":" << value.maximum << ",\"stddev\":"
      << value.stddev << '}';
  return out.str();
}

bool SameCellFamily(const OverlapCellResult &left,
                    const OverlapCellResult &right) noexcept {
  const auto &a = left.configuration;
  const auto &b = right.configuration;
  return a.workload == b.workload && a.direction == b.direction &&
         a.memory_class == b.memory_class &&
         std::abs(a.target_compute_us - b.target_compute_us) < 0.001;
}

double FitRate(const OverlapStatistics &statistics, double tolerance) {
  if (tolerance == 1.0)
    return statistics.fit_rate_1_percent;
  if (tolerance == 2.0)
    return statistics.fit_rate_2_percent;
  return statistics.fit_rate_5_percent;
}

struct ProfileKey {
  ComputeWorkload workload;
  double target_us;
  bool operator<(const ProfileKey &other) const noexcept {
    return std::tie(workload, target_us) <
           std::tie(other.workload, other.target_us);
  }
};

} // namespace

OverlapPlan DefaultOverlapPlan() {
  OverlapPlan plan;
  plan.workloads = {ComputeWorkload::SyntheticAlu,
                    ComputeWorkload::MemoryBound,
                    ComputeWorkload::Fp16Gemm};
  plan.compute_windows_us = {100, 250, 500, 1000, 2000,
                             4000, 8000, 16000, 32000, 64000};
  plan.h2d_sizes = {1 * kMiB,   2 * kMiB,   16 * kMiB, 64 * kMiB,
                    96 * kMiB,  128 * kMiB, 256 * kMiB, 512 * kMiB,
                    1024 * kMiB};
  plan.d2h_sizes = {2 * kMiB, 64 * kMiB, 128 * kMiB, 512 * kMiB};
  plan.memory_classes = {HostMemoryClass::PinnedHostAlloc};
  plan.maximum_host_bytes = 1024 * kMiB;
  plan.maximum_device_bytes = 1024 * kMiB + 768 * kMiB;
  return plan;
}

bool ShouldPruneOutsideFullHide(double c0_ns, double t0_ns,
                                double ratio) noexcept {
  return std::isfinite(c0_ns) && std::isfinite(t0_ns) && c0_ns > 0 &&
         t0_ns > c0_ns * ratio;
}

std::uint64_t
AdaptiveGateDelayNs(const std::vector<std::uint64_t> &submission_ns,
                    std::uint64_t minimum_delay_ns) noexcept {
  if (submission_ns.empty())
    return minimum_delay_ns;
  std::vector<std::uint64_t> sorted = submission_ns;
  std::sort(sorted.begin(), sorted.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(0.99 * static_cast<double>(sorted.size()))) - 1;
  const auto p99 = sorted[std::min(index, sorted.size() - 1)];
  if (p99 > std::numeric_limits<std::uint64_t>::max() / 4)
    return std::numeric_limits<std::uint64_t>::max();
  return std::max(minimum_delay_ns, p99 * 4);
}

OverlapMetrics CalculateOverlapMetrics(double c0_ns, double t0_ns,
                                       double cc_ns, double tc_ns,
                                       double makespan_ns) noexcept {
  OverlapMetrics value;
  if (!(c0_ns > 0) || !(t0_ns > 0) || !(cc_ns > 0) || !(tc_ns > 0) ||
      !(makespan_ns > 0) || !std::isfinite(c0_ns) || !std::isfinite(t0_ns) ||
      !std::isfinite(cc_ns) || !std::isfinite(tc_ns) ||
      !std::isfinite(makespan_ns))
    return value;
  value.critical_path_delta_ns = makespan_ns - std::max(c0_ns, t0_ns);
  value.critical_path_added_ns =
      std::max(0.0, value.critical_path_delta_ns);
  value.compute_slowdown = (cc_ns - c0_ns) / c0_ns;
  value.transfer_slowdown = (tc_ns - t0_ns) / t0_ns;
  value.overlap_efficiency_raw =
      (c0_ns + t0_ns - makespan_ns) / std::min(c0_ns, t0_ns);
  value.overlap_efficiency_normalized = Clamp01(value.overlap_efficiency_raw);
  value.compute_path_delta_ns = makespan_ns - c0_ns;
  value.compute_path_added_ns = std::max(0.0, value.compute_path_delta_ns);
  value.compute_path_added_percent =
      100.0 * value.compute_path_added_ns / c0_ns;
  value.hidden_fraction_raw = 1.0 - ((makespan_ns - c0_ns) / t0_ns);
  value.hidden_fraction_normalized = Clamp01(value.hidden_fraction_raw);
  value.compute_retention = c0_ns / cc_ns;
  value.fit_1_percent = value.compute_path_added_percent <= 1.0;
  value.fit_2_percent = value.compute_path_added_percent <= 2.0;
  value.fit_5_percent = value.compute_path_added_percent <= 5.0;
  return value;
}

OverlapStatistics
CalculateOverlapStatistics(const std::vector<OverlapSample> &samples) {
  OverlapStatistics value;
  value.valid_samples = static_cast<std::size_t>(std::count_if(
      samples.begin(), samples.end(), ValidForStatistics));
  value.c0_ns = DistributionOf(samples, [](const auto &s) {
    return static_cast<double>(s.c0_reference_ns);
  });
  value.t0_ns = DistributionOf(samples, [](const auto &s) {
    return static_cast<double>(s.t0_reference_ns);
  });
  value.cc_ns = DistributionOf(samples,
                               [](const auto &s) { return double(s.cc_ns); });
  value.tc_ns = DistributionOf(samples,
                               [](const auto &s) { return double(s.tc_ns); });
  value.makespan_ns = DistributionOf(samples, [](const auto &s) {
    return double(s.makespan_device_primary_ns);
  });
  value.added_ns = DistributionOf(samples, [](const auto &s) {
    return s.metrics.compute_path_added_ns;
  });
  value.added_percent = DistributionOf(samples, [](const auto &s) {
    return s.metrics.compute_path_added_percent;
  });
  value.compute_slowdown = DistributionOf(
      samples, [](const auto &s) { return s.metrics.compute_slowdown; });
  value.transfer_slowdown = DistributionOf(
      samples, [](const auto &s) { return s.metrics.transfer_slowdown; });
  value.hidden_fraction = DistributionOf(
      samples, [](const auto &s) { return s.metrics.hidden_fraction_raw; });
  value.primary_crosscheck_delta_ns = DistributionOf(samples, [](const auto &s) {
    return std::abs(double(s.makespan_device_primary_ns) -
                    double(s.makespan_device_crosscheck_ns));
  });
  if (value.valid_samples) {
    for (const auto &sample : samples) {
      if (!ValidForStatistics(sample))
        continue;
      value.fit_rate_1_percent += sample.metrics.fit_1_percent ? 1.0 : 0.0;
      value.fit_rate_2_percent += sample.metrics.fit_2_percent ? 1.0 : 0.0;
      value.fit_rate_5_percent += sample.metrics.fit_5_percent ? 1.0 : 0.0;
    }
    value.fit_rate_1_percent /= double(value.valid_samples);
    value.fit_rate_2_percent /= double(value.valid_samples);
    value.fit_rate_5_percent /= double(value.valid_samples);
  }
  value.p99_meaningful = value.valid_samples >= 100;
  return value;
}

std::vector<std::uint64_t>
SelectRefinementSizes(const std::vector<OverlapCellResult> &coarse_cells) {
  std::set<std::uint64_t> selected;
  for (std::size_t i = 0; i < coarse_cells.size(); ++i) {
    const auto &cell = coarse_cells[i];
    const auto p = cell.statistics.added_percent.p95;
    if ((p >= 0.5 && p <= 7.5) ||
        std::abs(cell.statistics.compute_slowdown.p95) >= 0.05 ||
        std::abs(cell.statistics.transfer_slowdown.p95) >= 0.10)
      selected.insert(cell.configuration.transfer_bytes);
    if (i && SameCellFamily(cell, coarse_cells[i - 1])) {
      const bool changed =
          (cell.statistics.added_percent.p95 <= 2.0) !=
          (coarse_cells[i - 1].statistics.added_percent.p95 <= 2.0);
      if (changed) {
        selected.insert(cell.configuration.transfer_bytes);
        selected.insert(coarse_cells[i - 1].configuration.transfer_bytes);
      }
    }
  }
  return {selected.begin(), selected.end()};
}

std::vector<SafeTransferEnvelope>
CalculateSafeEnvelopes(const std::vector<OverlapCellResult> &cells) {
  using Key = std::tuple<ComputeWorkload, TransferDirection, HostMemoryClass,
                         double, double>;
  std::map<Key, SafeTransferEnvelope> envelopes;
  for (const auto &cell : cells) {
    if (!cell.statistics.p99_meaningful ||
        (cell.status != OverlapStatus::Success &&
         cell.status != OverlapStatus::MeasurementSensitive))
      continue;
    for (const double tolerance : {1.0, 2.0, 5.0}) {
      if (cell.statistics.added_percent.p99 > tolerance)
        continue;
      const auto &c = cell.configuration;
      const Key key{c.workload, c.direction, c.memory_class,
                    c.target_compute_us, tolerance};
      auto &envelope = envelopes[key];
      if (c.transfer_bytes >= envelope.largest_measured_bytes) {
        envelope = {c.workload,
                    c.direction,
                    c.memory_class,
                    c.target_compute_us,
                    tolerance,
                    c.transfer_bytes,
                    cell.statistics.added_percent.p99,
                    FitRate(cell.statistics, tolerance),
                    c.target_compute_us <= 500 ? "MEASUREMENT_SENSITIVE"
                                               : "TRUSTED"};
      }
    }
  }
  std::vector<SafeTransferEnvelope> result;
  for (auto &[key, value] : envelopes) {
    (void)key;
    result.push_back(std::move(value));
  }
  return result;
}

OverlapRunReport RunOverlapLaboratory(IOverlapProvider &provider,
                                      const memory::MemorySnapshot &host,
                                      const OverlapRunOptions &options) {
  OverlapRunReport report;
  report.plan = options.plan;
  if (!provider.SupportsCuda()) {
    report.status = OverlapStatus::SkippedUnsupported;
    report.message = "CUDA overlap laboratory is unavailable in this build";
    return report;
  }
  report.device = provider.InspectDevice(options.device_index);
  if (!report.device.cuda_supported) {
    report.status = OverlapStatus::SkippedUnsupported;
    report.message = "CUDA device is unavailable";
    return report;
  }
  if (options.dry_run)
    return report;
  if ((options.run_coarse || options.run_refinement) &&
      std::find(options.plan.memory_classes.begin(),
                options.plan.memory_classes.end(),
                HostMemoryClass::PageablePretouched) !=
          options.plan.memory_classes.end()) {
    report.status = OverlapStatus::SkippedUnsupported;
    report.message =
        "pageable cudaMemcpyAsync may block host submission behind the "
        "pending release gate; pageable capacity-tier behavior is measured "
        "by the WU5 isolated controls, not the WU6 matched overlap topology";
    return report;
  }
  const auto host_decision = memory::PlanAllocation(
      host, options.host_safety, memory::MemoryMethod::CudaHostAlloc,
      options.plan.maximum_host_bytes);
  const auto device_decision = PlanDeviceMemory(
      report.device, options.plan.maximum_device_bytes, options.device_safety);
  if (!host_decision.safe() || !device_decision.safe) {
    report.status = OverlapStatus::SkippedSafetyLimit;
    report.message = host_decision.safe() ? device_decision.reason
                                          : host_decision.reason;
    return report;
  }

  const auto calibration_prepare = provider.Prepare(
      options.device_index, HostMemoryClass::PinnedHostAlloc,
      (!options.instrumentation_control && !options.run_coarse &&
       !options.run_refinement)
          ? std::min<std::uint64_t>(options.plan.maximum_host_bytes, 2 * kMiB)
          : (options.instrumentation_control && !options.run_coarse &&
                     !options.run_refinement
                 ? std::min<std::uint64_t>(options.plan.maximum_host_bytes,
                                           512 * kMiB)
                 : options.plan.maximum_host_bytes));
  if (!calibration_prepare.ok()) {
    report.status = OverlapStatus::CudaError;
    report.message = calibration_prepare.message;
    return report;
  }

  std::map<ProfileKey, WorkloadProfile> profiles;
  std::vector<std::uint64_t> submissions;
  if (options.calibrate) {
    for (const auto workload : options.plan.workloads)
      for (const auto target : options.plan.compute_windows_us) {
        auto profile = provider.Calibrate(
            workload, target, options.plan.warmups,
            options.plan.calibration_repetitions);
        if (!profile.validated && report.status == OverlapStatus::Success) {
          report.status = profile.status;
          report.message = profile.message.empty()
                               ? "compute workload calibration validation failed"
                               : profile.message;
        }
        profiles[{workload, target}] = profile;
        report.profiles.push_back(std::move(profile));
      }
  }

  if (report.status != OverlapStatus::Success) {
    (void)provider.Release();
    return report;
  }

  if (options.instrumentation_control) {
    const std::set<double> selected_windows{100, 1000, 4000, 16000};
    const std::set<std::uint64_t> selected_sizes{2 * kMiB, 64 * kMiB,
                                                 128 * kMiB, 512 * kMiB};
    constexpr std::uint32_t repetitions = 31;
    for (const auto workload : options.plan.workloads) {
      for (const auto target : options.plan.compute_windows_us) {
        if (!selected_windows.contains(target))
          continue;
        const auto found = profiles.find({workload, target});
        if (found == profiles.end() || !found->second.validated)
          continue;
        for (const auto bytes : options.plan.h2d_sizes) {
          if (!selected_sizes.contains(bytes))
            continue;
          constexpr std::array modes{InstrumentationMode::Minimal,
                                     InstrumentationMode::MatchedGateDependency,
                                     InstrumentationMode::Full};
          struct ControlSamples {
            std::vector<double> compute, transfer, host_time,
                topology_overhead;
            bool invalid{false};
          } samples[3];

          // Untimed control warmups establish comparable device state before
          // the rotated A/B/C sequence begins.
          for (std::size_t mode_index = 0; mode_index < modes.size();
               ++mode_index) {
            const auto gate = AdaptiveGateDelayNs(submissions);
            (void)provider.RunTopology(
                found->second, TransferDirection::H2D, bytes,
                SampleMode::ComputeOnly, modes[mode_index], gate, 250'000, 0);
            (void)provider.RunTopology(
                found->second, TransferDirection::H2D, bytes,
                SampleMode::TransferOnly, modes[mode_index], gate, 250'000, 0);
          }
          for (std::uint32_t repetition = 0; repetition < repetitions;
               ++repetition) {
            for (std::size_t order = 0; order < modes.size(); ++order) {
              const auto mode_index = (repetition + order) % modes.size();
              const auto mode = modes[mode_index];
              const auto gate = AdaptiveGateDelayNs(submissions);
              const auto c = provider.RunTopology(
                  found->second, TransferDirection::H2D, bytes,
                  SampleMode::ComputeOnly, mode, gate, 250'000, repetition);
              const auto t = provider.RunTopology(
                  found->second, TransferDirection::H2D, bytes,
                  SampleMode::TransferOnly, mode, gate, 250'000, repetition);
              submissions.push_back(
                  std::max(c.host_submission_ns, t.host_submission_ns));
              if (c.status == OverlapStatus::Success &&
                  t.status == OverlapStatus::Success) {
                samples[mode_index].compute.push_back(double(c.compute_ns));
                samples[mode_index].transfer.push_back(double(t.transfer_ns));
                samples[mode_index].host_time.push_back(
                    double(c.makespan_host_ns + t.makespan_host_ns));
                samples[mode_index].topology_overhead.push_back(
                    std::max(0.0, double(c.makespan_device_primary_ns) -
                                      double(c.compute_ns)));
                samples[mode_index].topology_overhead.push_back(
                    std::max(0.0, double(t.makespan_device_primary_ns) -
                                      double(t.transfer_ns)));
              } else
                samples[mode_index].invalid = true;
            }
          }

          std::array<InstrumentationObservation, 3> observations;
          std::array<double, 3> paired_compute_bias{};
          std::array<double, 3> paired_transfer_bias{};
          std::array<bool, 3> paired_compute_significant{};
          std::array<bool, 3> paired_transfer_significant{};
          for (std::size_t mode_index = 1; mode_index < modes.size();
               ++mode_index) {
            std::vector<double> compute_bias, transfer_bias;
            // A->B measures the cost of introducing the mandatory pending
            // gate/dependency boundary. B->C is the authoritative observer
            // qualification because all WU6 baselines and concurrent samples
            // use the gate and differ only by the full timing/JOIN topology.
            const auto reference_index = mode_index - 1;
            const auto count = std::min(
                {samples[reference_index].compute.size(),
                 samples[mode_index].compute.size(),
                 samples[reference_index].transfer.size(),
                 samples[mode_index].transfer.size()});
            for (std::size_t sample = 0; sample < count; ++sample) {
              if (samples[reference_index].compute[sample] > 0)
                compute_bias.push_back(
                    100.0 * (samples[mode_index].compute[sample] -
                             samples[reference_index].compute[sample]) /
                    samples[reference_index].compute[sample]);
              if (samples[reference_index].transfer[sample] > 0)
                transfer_bias.push_back(
                    100.0 * (samples[mode_index].transfer[sample] -
                             samples[reference_index].transfer[sample]) /
                    samples[reference_index].transfer[sample]);
            }
            paired_compute_significant[mode_index] =
                PairedDirectionIsSignificant(compute_bias);
            paired_transfer_significant[mode_index] =
                PairedDirectionIsSignificant(transfer_bias);
            paired_compute_bias[mode_index] =
                trace::CalculateDistribution(std::move(compute_bias)).median;
            paired_transfer_bias[mode_index] =
                trace::CalculateDistribution(std::move(transfer_bias)).median;
          }
          for (std::size_t mode_index = 0; mode_index < modes.size();
               ++mode_index) {
            auto &observation = observations[mode_index];
            observation.workload = workload;
            observation.direction = TransferDirection::H2D;
            observation.transfer_bytes = bytes;
            observation.target_compute_us = target;
            observation.mode = modes[mode_index];
            observation.compute_ns =
                trace::CalculateDistribution(
                    std::move(samples[mode_index].compute));
            observation.transfer_ns =
                trace::CalculateDistribution(
                    std::move(samples[mode_index].transfer));
            observation.host_ns =
                trace::CalculateDistribution(
                    std::move(samples[mode_index].host_time));
            observation.topology_overhead_ns =
                trace::CalculateDistribution(
                    std::move(samples[mode_index].topology_overhead));
            if (samples[mode_index].invalid || observation.compute_ns.mean <= 0 ||
                observation.transfer_ns.mean <= 0) {
              observation.rejected = true;
              observation.status = OverlapStatus::InvalidGate;
              report.status = OverlapStatus::InvalidGate;
              report.message =
                  "instrumentation control did not produce valid gated samples";
            }
          }
          for (std::size_t mode_index = 1; mode_index < modes.size();
               ++mode_index) {
            auto &observation = observations[mode_index];
            if (observation.status != OverlapStatus::Success)
              continue;
            const auto &reference = observations[mode_index - 1];
            observation.compute_bias_percent =
                paired_compute_bias[mode_index];
            observation.transfer_bias_percent =
                paired_transfer_bias[mode_index];
            const bool compute_material =
                std::abs(observation.compute_bias_percent) > 2.0 &&
                std::abs(observation.compute_bias_percent) *
                        reference.compute_ns.median /
                        100.0 >
                    5'000.0;
            const bool transfer_material =
                std::abs(observation.transfer_bias_percent) > 2.0 &&
                std::abs(observation.transfer_bias_percent) *
                        reference.transfer_ns.median / 100.0 >
                    5'000.0;
            const bool compute_sensitive =
                reference.compute_ns.median <= 500'000.0;
            const bool transfer_sensitive =
                reference.transfer_ns.median <= 500'000.0;
            // B is retained as a diagnostic measurement of the common gate's
            // effect relative to minimal timing. The qualification gate is C
            // relative to B: authoritative C0/T0/concurrent samples all use C,
            // so A is not a structurally valid authority baseline.
            observation.rejected =
                mode_index == 2 &&
                ((compute_material && paired_compute_significant[mode_index] &&
                  !compute_sensitive) ||
                 (transfer_material &&
                  paired_transfer_significant[mode_index] &&
                  !transfer_sensitive));
            if (observation.rejected) {
              observation.status = OverlapStatus::InstrumentationBias;
              report.status = OverlapStatus::InstrumentationBias;
              report.message =
                  "instrumentation control detected material observer bias";
            } else if ((compute_material &&
                        (compute_sensitive ||
                         (mode_index == 2 &&
                          !paired_compute_significant[mode_index]))) ||
                       (transfer_material &&
                        (transfer_sensitive ||
                         (mode_index == 2 &&
                          !paired_transfer_significant[mode_index]))))
              observation.status = OverlapStatus::MeasurementSensitive;
          }
          for (auto &observation : observations)
            report.instrumentation.push_back(std::move(observation));
        }
      }
    }
    if (report.status != OverlapStatus::Success) {
      (void)provider.Release();
      return report;
    }
  }

  std::uint64_t baseline_block = 0;
  const auto run_backend = [&](HostMemoryClass backend,
                               TransferDirection direction,
                               const std::vector<std::uint64_t> &sizes,
                               OverlapPhase phase,
                               std::uint32_t repetitions,
                               const std::set<std::uint64_t> *selected) {
    const auto prepared = provider.Prepare(options.device_index, backend,
                                           options.plan.maximum_host_bytes);
    if (!prepared.ok()) {
      report.status = OverlapStatus::CudaError;
      report.message = prepared.message;
      return false;
    }
    for (const auto workload : options.plan.workloads) {
      for (const auto target : options.plan.compute_windows_us) {
        auto found = profiles.find({workload, target});
        if (found == profiles.end() || !found->second.validated)
          continue;
        for (const auto bytes : sizes) {
          if (selected && !selected->contains(bytes))
            continue;
          if (bytes > options.plan.maximum_host_bytes)
            continue;
          if (bytes == 1024 * kMiB && target != 1000 && target < 32000)
            continue;
          OverlapCellResult cell;
          cell.configuration = {workload,
                                direction,
                                backend,
                                bytes,
                                target,
                                phase,
                                repetitions,
                                0x5349444543415236ULL,
                                AdaptiveGateDelayNs(submissions),
                                250'000,
                                options.device_index};
          cell.profile = found->second;
          cell.refined = phase == OverlapPhase::Refine ||
                         phase == OverlapPhase::Repeat;
          if (cell.refined)
            cell.refinement_reason =
                "P95/FIT/SLOWDOWN_BOUNDARY_SELECTED_FROM_COARSE_PHASE";
          cell.telemetry_before = provider.ReadTelemetry(options.device_index);
          for (std::uint32_t repetition = 0;
               repetition < cell.configuration.repetitions; ++repetition) {
            const auto gate_delay = AdaptiveGateDelayNs(submissions);
            const auto c0 = provider.RunTopology(
                cell.profile, direction, bytes, SampleMode::ComputeOnly,
                InstrumentationMode::Full, gate_delay,
                cell.configuration.gate_margin_ns, repetition);
            const auto t0 = provider.RunTopology(
                cell.profile, direction, bytes, SampleMode::TransferOnly,
                InstrumentationMode::Full, gate_delay,
                cell.configuration.gate_margin_ns, repetition);
            const auto concurrent = provider.RunTopology(
                cell.profile, direction, bytes, SampleMode::Concurrent,
                InstrumentationMode::Full, gate_delay,
                cell.configuration.gate_margin_ns, repetition);
            submissions.push_back(std::max(
                {c0.host_submission_ns, t0.host_submission_ns,
                 concurrent.host_submission_ns}));
            if (submissions.size() > 256)
              submissions.erase(submissions.begin(), submissions.begin() + 128);
            OverlapSample sample;
            sample.repetition = repetition;
            sample.baseline_block_id = ++baseline_block;
            sample.c0_reference_ns = c0.compute_ns;
            sample.t0_reference_ns = t0.transfer_ns;
            sample.cc_ns = concurrent.compute_ns;
            sample.tc_ns = concurrent.transfer_ns;
            sample.makespan_device_primary_ns =
                concurrent.makespan_device_primary_ns;
            sample.makespan_device_crosscheck_ns =
                concurrent.makespan_device_crosscheck_ns;
            sample.makespan_host_ns = concurrent.makespan_host_ns;
            sample.host_submission_ns = concurrent.host_submission_ns;
            sample.gate_delay_ns = gate_delay;
            sample.gate_actual_ns = concurrent.gate_actual_ns;
            sample.gate_margin_ns = cell.configuration.gate_margin_ns;
            sample.gate_valid = c0.gate_valid && t0.gate_valid &&
                                concurrent.gate_valid;
            sample.native_error = concurrent.native_error;
            sample.status = concurrent.status;
            sample.message = concurrent.message;
            if (!sample.gate_valid)
              sample.status = OverlapStatus::InvalidGate;
            else if (c0.status != OverlapStatus::Success)
              sample.status = c0.status;
            else if (t0.status != OverlapStatus::Success)
              sample.status = t0.status;
            else if (target <= 500)
              sample.status = OverlapStatus::MeasurementSensitive;
            sample.metrics = CalculateOverlapMetrics(
                double(sample.c0_reference_ns), double(sample.t0_reference_ns),
                double(sample.cc_ns), double(sample.tc_ns),
                double(sample.makespan_device_primary_ns));
            if (!sample.c0_reference_ns || !sample.t0_reference_ns ||
                !sample.cc_ns || !sample.tc_ns ||
                !sample.makespan_device_primary_ns)
              sample.status = OverlapStatus::ImpossibleDuration;
            else if (sample.makespan_device_primary_ns <
                     std::max(sample.cc_ns, sample.tc_ns))
              sample.status = OverlapStatus::InvalidEventOrder;
            else if (!std::isfinite(sample.metrics.compute_path_added_percent) ||
                     !std::isfinite(sample.metrics.compute_slowdown) ||
                     !std::isfinite(sample.metrics.transfer_slowdown) ||
                     !std::isfinite(sample.metrics.hidden_fraction_raw) ||
                     !std::isfinite(sample.metrics.overlap_efficiency_raw))
              sample.status = OverlapStatus::NanOrInf;
            cell.samples.push_back(std::move(sample));
            if (repetition >= 2 &&
                ShouldPruneOutsideFullHide(
                    cell.statistics.c0_ns.median > 0
                        ? cell.statistics.c0_ns.median
                        : double(c0.compute_ns),
                    cell.statistics.t0_ns.median > 0
                        ? cell.statistics.t0_ns.median
                        : double(t0.transfer_ns)) &&
                phase == OverlapPhase::Coarse)
              break;
          }
          cell.statistics = CalculateOverlapStatistics(cell.samples);
          cell.telemetry_after = provider.ReadTelemetry(options.device_index);
          if (cell.statistics.valid_samples == 0)
            cell.status = cell.samples.empty() ? OverlapStatus::InternalError
                                               : cell.samples.front().status;
          else if (cell.samples.size() < cell.configuration.repetitions &&
                   ShouldPruneOutsideFullHide(cell.statistics.c0_ns.median,
                                              cell.statistics.t0_ns.median))
            cell.status = OverlapStatus::OutsideFullHideRegion;
          else if (target <= 500)
            cell.status = OverlapStatus::MeasurementSensitive;
          if (cell.statistics.valid_samples >= 4) {
            std::vector<double> first_half, second_half;
            const auto midpoint = cell.samples.size() / 2;
            for (std::size_t i = 0; i < cell.samples.size(); ++i) {
              if (!ValidForStatistics(cell.samples[i]))
                continue;
              (i < midpoint ? first_half : second_half)
                  .push_back(double(cell.samples[i].c0_reference_ns));
            }
            const auto first =
                trace::CalculateDistribution(std::move(first_half)).median;
            const auto second =
                trace::CalculateDistribution(std::move(second_half)).median;
            if (first > 0 && std::abs(second - first) > 10'000.0 &&
                std::abs(second - first) / first > 0.10)
              cell.status = OverlapStatus::BaselineDrift;
            else if (cell.statistics.c0_ns.mean > 0 &&
                     cell.statistics.c0_ns.stddev > 10'000.0 &&
                     cell.statistics.c0_ns.stddev /
                             cell.statistics.c0_ns.mean >
                         0.15)
              cell.status = OverlapStatus::Noisy;
          }
          report.cells.push_back(std::move(cell));
        }
      }
    }
    const auto validation = provider.Validate(
        options.plan.workloads.empty() || options.plan.compute_windows_us.empty()
            ? WorkloadProfile{}
            : profiles[{options.plan.workloads.front(),
                        options.plan.compute_windows_us.front()}],
        direction, sizes.empty() ? 0 : sizes.back());
    if (!validation.ok()) {
      report.status = OverlapStatus::TransferValidationFailure;
      report.message = validation.message;
      return false;
    }
    return true;
  };

  if (options.run_coarse) {
    for (const auto backend : options.plan.memory_classes) {
      if (!options.plan.h2d_sizes.empty() &&
          !run_backend(backend, TransferDirection::H2D,
                       options.plan.h2d_sizes, OverlapPhase::Coarse,
                       options.plan.coarse_repetitions, nullptr))
        break;
      if (report.status == OverlapStatus::Success &&
          !options.plan.d2h_sizes.empty() &&
          !run_backend(backend, TransferDirection::D2H,
                       options.plan.d2h_sizes, OverlapPhase::Coarse,
                       options.plan.coarse_repetitions, nullptr))
        break;
    }
  }
  if (options.run_refinement && report.status == OverlapStatus::Success) {
    const auto selected_values = SelectRefinementSizes(report.cells);
    const std::set<std::uint64_t> selected(selected_values.begin(),
                                           selected_values.end());
    for (const auto backend : options.plan.memory_classes) {
      if (!options.plan.h2d_sizes.empty() &&
          !run_backend(backend, TransferDirection::H2D,
                       options.plan.h2d_sizes, OverlapPhase::Refine,
                       options.plan.refine_repetitions, &selected))
        break;
      if (report.status == OverlapStatus::Success &&
          !options.plan.d2h_sizes.empty() &&
          !run_backend(backend, TransferDirection::D2H,
                       options.plan.d2h_sizes, OverlapPhase::Refine,
                       options.plan.refine_repetitions, &selected))
        break;
    }
  }
  report.envelopes = CalculateSafeEnvelopes(report.cells);
  const auto released = provider.Release();
  if (!released.ok() && report.status == OverlapStatus::Success) {
    report.status = OverlapStatus::CudaError;
    report.message = released.message;
  }
  return report;
}

std::string
OverlapConfigurationIdentity(const OverlapConfiguration &configuration) {
  std::ostringstream out;
  out << "workload=" << ToString(configuration.workload)
      << ";direction=" << ToString(configuration.direction)
      << ";memory=" << ToString(configuration.memory_class)
      << ";bytes=" << configuration.transfer_bytes
      << ";target_us=" << std::fixed << std::setprecision(3)
      << configuration.target_compute_us
      << ";phase=" << ToString(configuration.phase)
      << ";device=" << configuration.device_index;
  return core::Sha256Hex(out.str());
}

std::string OverlapRunReportToJson(const OverlapRunReport &report) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"schema\":\"sidecar.cuda-overlap.v1\""
      << ",\"status\":\"" << ToString(report.status) << "\""
      << ",\"message\":\"" << Escape(report.message) << "\""
      << ",\"device\":{\"index\":" << report.device.device_index
      << ",\"name\":\"" << Escape(report.device.name)
      << "\",\"free_bytes\":" << report.device.free_bytes
      << ",\"total_bytes\":" << report.device.total_bytes << "}"
      << ",\"profiles\":[";
  for (std::size_t i = 0; i < report.profiles.size(); ++i) {
    if (i)
      out << ',';
    const auto &p = report.profiles[i];
    out << "{\"workload\":\"" << ToString(p.workload)
        << "\",\"target_us\":" << p.target_us
        << ",\"calibrated_us\":" << DistributionJson(p.calibrated_us)
        << ",\"alu_iterations\":" << p.alu_iterations
        << ",\"memory_working_set_bytes\":" << p.memory_working_set_bytes
        << ",\"memory_passes\":" << p.memory_passes
        << ",\"gemm\":{\"m\":" << p.gemm_m << ",\"n\":" << p.gemm_n
        << ",\"k\":" << p.gemm_k << ",\"repetitions\":"
        << p.gemm_repetitions << "},\"validated\":"
        << (p.validated ? "true" : "false") << ",\"status\":\""
        << ToString(p.status) << "\"}";
  }
  out << "],\"instrumentation\":[";
  for (std::size_t i = 0; i < report.instrumentation.size(); ++i) {
    if (i)
      out << ',';
    const auto &c = report.instrumentation[i];
    out << "{\"workload\":\"" << ToString(c.workload)
        << "\",\"direction\":\"" << ToString(c.direction)
        << "\",\"transfer_bytes\":" << c.transfer_bytes
        << ",\"target_compute_us\":" << c.target_compute_us
        << ",\"mode\":\"" << ToString(c.mode)
        << "\",\"compute_ns\":" << DistributionJson(c.compute_ns)
        << ",\"transfer_ns\":" << DistributionJson(c.transfer_ns)
        << ",\"host_ns\":" << DistributionJson(c.host_ns)
        << ",\"topology_overhead_ns\":"
        << DistributionJson(c.topology_overhead_ns)
        << ",\"compute_bias_percent\":" << c.compute_bias_percent
        << ",\"transfer_bias_percent\":" << c.transfer_bias_percent
        << ",\"rejected\":" << (c.rejected ? "true" : "false")
        << ",\"status\":\"" << ToString(c.status) << "\"}";
  }
  out << "],\"cells\":[";
  for (std::size_t i = 0; i < report.cells.size(); ++i) {
    if (i)
      out << ',';
    const auto &c = report.cells[i];
    out << "{\"configuration\":\""
        << Escape(OverlapConfigurationIdentity(c.configuration))
        << "\",\"status\":\"" << ToString(c.status)
        << "\",\"valid_samples\":" << c.statistics.valid_samples
        << ",\"c0_ns\":" << DistributionJson(c.statistics.c0_ns)
        << ",\"t0_ns\":" << DistributionJson(c.statistics.t0_ns)
        << ",\"added_percent\":"
        << DistributionJson(c.statistics.added_percent)
        << ",\"fit_rate_1pct\":" << c.statistics.fit_rate_1_percent
        << ",\"fit_rate_2pct\":" << c.statistics.fit_rate_2_percent
        << ",\"fit_rate_5pct\":" << c.statistics.fit_rate_5_percent
        << ",\"p99_meaningful\":"
        << (c.statistics.p99_meaningful ? "true" : "false") << '}';
  }
  out << "],\"envelopes\":[";
  for (std::size_t i = 0; i < report.envelopes.size(); ++i) {
    if (i)
      out << ',';
    const auto &e = report.envelopes[i];
    out << "{\"workload\":\"" << ToString(e.workload)
        << "\",\"direction\":\"" << ToString(e.direction)
        << "\",\"memory\":\"" << ToString(e.memory_class)
        << "\",\"target_us\":" << e.target_compute_us
        << ",\"tolerance_percent\":" << e.tolerance_percent
        << ",\"largest_measured_bytes\":" << e.largest_measured_bytes
        << ",\"p99_added_percent\":" << e.p99_added_percent
        << ",\"fit_rate\":" << e.fit_rate << ",\"confidence\":\""
        << e.confidence << "\"}";
  }
  out << "]}";
  return out.str();
}

std::string FormatOverlapRunReport(const OverlapRunReport &report) {
  std::ostringstream out;
  out << "SIDECAR CUDA COMPUTE + TRANSFER OVERLAP\n\nStatus: "
      << ToString(report.status) << '\n';
  if (!report.message.empty())
    out << "Message: " << report.message << '\n';
  out << "Device: " << (report.device.name.empty() ? "unknown" : report.device.name)
      << "\nProfiles: " << report.profiles.size()
      << "\nInstrumentation controls: " << report.instrumentation.size()
      << "\nMeasured cells: " << report.cells.size()
      << "\nSafe envelopes: " << report.envelopes.size() << "\n\n";
  for (const auto &control : report.instrumentation)
    out << "Instrumentation " << ToString(control.workload) << ' '
        << control.target_compute_us << " us "
        << double(control.transfer_bytes) / double(kMiB) << " MiB "
        << ToString(control.mode) << " compute_bias=" << std::fixed
        << std::setprecision(3) << control.compute_bias_percent
        << "% transfer_bias=" << control.transfer_bias_percent
        << "% topology_overhead_median_ns="
        << control.topology_overhead_ns.median << " status="
        << ToString(control.status) << '\n';
  if (!report.instrumentation.empty())
    out << '\n';
  out << "Workload       Direction Backend          Window(us) Bytes(MiB) P99 added(%) Fit<=2% Status\n";
  for (const auto &cell : report.cells) {
    const auto &c = cell.configuration;
    out << std::left << std::setw(14) << ToString(c.workload) << std::setw(10)
        << ToString(c.direction) << std::setw(17) << ToString(c.memory_class)
        << std::right << std::setw(11) << c.target_compute_us << std::setw(11)
        << double(c.transfer_bytes) / double(kMiB) << std::setw(13)
        << std::fixed << std::setprecision(3)
        << cell.statistics.added_percent.p99 << std::setw(9)
        << cell.statistics.fit_rate_2_percent * 100.0 << "% "
        << ToString(cell.status) << '\n';
  }
  return out.str();
}

const char *ToString(ComputeWorkload value) noexcept {
  switch (value) {
  case ComputeWorkload::SyntheticAlu:
    return "SYNTHETIC_ALU";
  case ComputeWorkload::MemoryBound:
    return "MEMORY_BOUND";
  case ComputeWorkload::Fp16Gemm:
    return "FP16_GEMM";
  }
  return "UNKNOWN";
}
const char *ToString(OverlapPhase value) noexcept {
  switch (value) {
  case OverlapPhase::Calibration:
    return "CALIBRATION";
  case OverlapPhase::Instrumentation:
    return "INSTRUMENTATION";
  case OverlapPhase::Coarse:
    return "COARSE";
  case OverlapPhase::Refine:
    return "REFINE";
  case OverlapPhase::Repeat:
    return "REPEAT";
  }
  return "UNKNOWN";
}
const char *ToString(SampleMode value) noexcept {
  switch (value) {
  case SampleMode::ComputeOnly:
    return "COMPUTE_ONLY";
  case SampleMode::TransferOnly:
    return "TRANSFER_ONLY";
  case SampleMode::Concurrent:
    return "CONCURRENT";
  }
  return "UNKNOWN";
}
const char *ToString(InstrumentationMode value) noexcept {
  switch (value) {
  case InstrumentationMode::Minimal:
    return "MINIMAL";
  case InstrumentationMode::MatchedGateDependency:
    return "MATCHED_GATE_DEPENDENCY";
  case InstrumentationMode::Full:
    return "FULL";
  }
  return "UNKNOWN";
}
const char *ToString(OverlapStatus value) noexcept {
  switch (value) {
  case OverlapStatus::Success:
    return "SUCCESS";
  case OverlapStatus::SkippedUnsupported:
    return "SKIPPED_UNSUPPORTED";
  case OverlapStatus::SkippedSafetyLimit:
    return "SKIPPED_SAFETY_LIMIT";
  case OverlapStatus::OutsideFullHideRegion:
    return "OUTSIDE_FULL_HIDE_REGION";
  case OverlapStatus::InvalidGate:
    return "INVALID_GATE";
  case OverlapStatus::InstrumentationBias:
    return "INSTRUMENTATION_BIAS";
  case OverlapStatus::ComputeValidationFailure:
    return "COMPUTE_VALIDATION_FAILURE";
  case OverlapStatus::TransferValidationFailure:
    return "TRANSFER_VALIDATION_FAILURE";
  case OverlapStatus::CudaError:
    return "CUDA_ERROR";
  case OverlapStatus::CublasError:
    return "CUBLAS_ERROR";
  case OverlapStatus::Noisy:
    return "NOISY";
  case OverlapStatus::InvalidEventOrder:
    return "INVALID_EVENT_ORDER";
  case OverlapStatus::BaselineDrift:
    return "BASELINE_DRIFT";
  case OverlapStatus::MeasurementSensitive:
    return "MEASUREMENT_SENSITIVE";
  case OverlapStatus::NanOrInf:
    return "NAN_OR_INF";
  case OverlapStatus::ImpossibleDuration:
    return "IMPOSSIBLE_DURATION";
  case OverlapStatus::InternalError:
    return "INTERNAL_ERROR";
  }
  return "UNKNOWN";
}

} // namespace sidecar::cuda
