#include "sidecar/memory/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace sidecar::memory {
namespace {

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

PhaseTiming Measure(const TimerCalibration& calibration, const auto& operation,
                    ProviderResult& result) {
    const auto start = NowNs();
    result = operation();
    const auto raw = NowNs() - start;
    return {raw, raw > calibration.bracket_overhead_ns
                     ? raw - calibration.bracket_overhead_ns : 0};
}

std::int64_t SignedDifference(std::optional<std::uint64_t> after,
                              std::optional<std::uint64_t> before) {
    if (!after || !before) return 0;
    if (*after >= *before) {
        const auto delta = *after - *before;
        return delta > static_cast<std::uint64_t>(INT64_MAX) ? INT64_MAX
                                                             : static_cast<std::int64_t>(delta);
    }
    const auto delta = *before - *after;
    return delta > static_cast<std::uint64_t>(INT64_MAX) ? INT64_MIN
                                                         : -static_cast<std::int64_t>(delta);
}

struct RegionCleanup {
    IHostMemoryProvider* provider{nullptr};
    MemoryMethod method{MemoryMethod::Pageable};
    MemoryRegion* region{nullptr};
    bool registered{false};
    ~RegionCleanup() {
        if (!provider || !region || !region->data) return;
        if (registered) {
            const auto result = provider->UnregisterCudaHost(*region);
            if (!result.ok()) return;
            registered = false;
        }
        if (method == MemoryMethod::CudaHostAlloc)
            (void)provider->FreeCudaHost(*region);
        else
            (void)provider->FreePageable(*region);
    }
};

bool ShouldStopAfter(ResultStatus status) {
    return status == ResultStatus::CudaOutOfMemory ||
           status == ResultStatus::CudaHostMemoryError ||
           status == ResultStatus::OsAllocationFailure ||
           status == ResultStatus::RegistrationFailure ||
           status == ResultStatus::CleanupFailure ||
           status == ResultStatus::AbortedPressureSignal;
}

LifecycleSample RunOne(IHostMemoryProvider& provider,
                       MemoryMethod method,
                       RegistrationMode mode,
                       std::uint64_t bytes,
                       std::uint32_t repetition,
                       bool cold,
                       const TimerCalibration& timer,
                       const SafetyDecision& safety,
                       const LifecycleOptions& options) {
    LifecycleSample sample;
    sample.method = method;
    sample.mode = mode;
    sample.requested_bytes = bytes;
    sample.repetition = repetition;
    sample.cold_setup = cold;
    sample.safety = safety;
    sample.before = provider.Snapshot();
    MemoryRegion region;
    RegionCleanup guard{&provider, method, &region, false};
    ProviderResult phase;

    auto allocate = [&]() {
        return method == MemoryMethod::CudaHostAlloc
                   ? provider.AllocateCudaHost(bytes, region)
                   : provider.AllocatePageable(bytes, region);
    };
    sample.backing_allocation = Measure(timer, allocate, phase);
    if (!phase.ok()) {
        sample.status = phase.status;
        sample.message = phase.message;
        if (method == MemoryMethod::Pageable || method == MemoryMethod::CudaHostRegister)
            sample.windows_error = phase.native_error;
        else
            sample.cuda_error = phase.native_error;
        sample.after_cleanup = provider.Snapshot();
        return sample;
    }

    std::uint64_t checksum = 0;
    if (method == MemoryMethod::CudaHostRegister && mode == RegistrationMode::Pretouched) {
        sample.first_touch = Measure(timer, [&] { return provider.Touch(region, 0, checksum); }, phase);
        if (!phase.ok()) sample.status = phase.status;
    }

    if (sample.status == ResultStatus::Success && method == MemoryMethod::CudaHostRegister) {
        sample.registration = Measure(timer, [&] { return provider.RegisterCudaHost(region); }, phase);
        if (!phase.ok()) {
            sample.status = phase.status;
            sample.cuda_error = phase.native_error;
            sample.message = phase.message;
        } else {
            guard.registered = true;
        }
    }

    if (sample.status == ResultStatus::Success &&
        !(method == MemoryMethod::CudaHostRegister && mode == RegistrationMode::Pretouched)) {
        sample.first_touch = Measure(timer, [&] { return provider.Touch(region, 0, checksum); }, phase);
        if (!phase.ok()) sample.status = phase.status;
    }

    if (sample.status == ResultStatus::Success) {
        sample.warm_touch = Measure(timer, [&] { return provider.Touch(region, 1, checksum); }, phase);
        if (!phase.ok()) sample.status = phase.status;
    }
    sample.after_setup = provider.Snapshot();

    if (method == MemoryMethod::CudaHostRegister && guard.registered) {
        sample.unregistration = Measure(timer, [&] { return provider.UnregisterCudaHost(region); }, phase);
        if (!phase.ok()) {
            sample.status = ResultStatus::CleanupFailure;
            sample.cuda_error = phase.native_error;
            sample.message = phase.message;
            // One untimed recovery attempt is allowed.  The failed timed operation
            // remains the reported result, but backing storage is never released
            // while CUDA may still consider it registered.
            if (provider.UnregisterCudaHost(region).ok()) guard.registered = false;
        } else {
            guard.registered = false;
        }
    }

    if (!guard.registered) {
        sample.cleanup = Measure(timer, [&] {
            return method == MemoryMethod::CudaHostAlloc
                       ? provider.FreeCudaHost(region) : provider.FreePageable(region);
        }, phase);
        if (!phase.ok()) {
            sample.status = ResultStatus::CleanupFailure;
            sample.message = phase.message;
            if (method == MemoryMethod::Pageable || method == MemoryMethod::CudaHostRegister)
                sample.windows_error = phase.native_error;
            else
                sample.cuda_error = phase.native_error;
        }
    }

    for (std::uint32_t check = 0; check < options.cooldown_checks; ++check) {
        sample.after_cleanup = provider.Snapshot();
        const auto delta = SignedDifference(sample.after_cleanup.available_physical_bytes,
                                            sample.before.available_physical_bytes);
        if (delta >= 0 || static_cast<std::uint64_t>(-delta) <=
                              std::max(options.safety.cleanup_warning_bytes, bytes / 10))
            break;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(options.cooldown_milliseconds));
    }
    sample.available_recovery_delta_bytes = SignedDifference(
        sample.after_cleanup.available_physical_bytes, sample.before.available_physical_bytes);
    sample.available_recovery_percent = bytes == 0 ? 0.0
        : 100.0 * static_cast<double>(sample.available_recovery_delta_bytes) /
              static_cast<double>(bytes);
    const auto lost = sample.available_recovery_delta_bytes < 0
                          ? static_cast<std::uint64_t>(-sample.available_recovery_delta_bytes) : 0;
    sample.noisy = lost > std::max(options.safety.cleanup_warning_bytes, bytes / 10);
    if (lost > std::max(options.safety.cleanup_hard_stop_bytes, bytes / 4)) {
        sample.status = ResultStatus::AbortedPressureSignal;
        sample.message = "cleanup did not restore acceptable physical-memory headroom";
    }
    return sample;
}

LifecycleResult Aggregate(MemoryMethod method, RegistrationMode mode,
                          std::uint64_t bytes, std::vector<LifecycleSample> samples) {
    LifecycleResult result;
    result.method = method;
    result.mode = mode;
    result.requested_bytes = bytes;
    result.samples = std::move(samples);
    result.status = ResultStatus::Success;
    for (const auto& sample : result.samples) {
        if (sample.status != ResultStatus::Success) {
            result.status = sample.status;
            result.message = sample.message;
            break;
        }
    }
    result.allocation = CalculatePhaseStatistics(result.samples,
                                                  &LifecycleSample::backing_allocation);
    result.first_touch = CalculatePhaseStatistics(result.samples, &LifecycleSample::first_touch);
    result.warm_touch = CalculatePhaseStatistics(result.samples, &LifecycleSample::warm_touch);
    result.registration = CalculatePhaseStatistics(result.samples, &LifecycleSample::registration);
    result.unregistration = CalculatePhaseStatistics(result.samples,
                                                      &LifecycleSample::unregistration);
    result.cleanup = CalculatePhaseStatistics(result.samples, &LifecycleSample::cleanup);
    const double setup_ns = result.allocation.corrected_ns.median +
                            result.first_touch.corrected_ns.median +
                            result.registration.corrected_ns.median;
    for (const auto reuse : {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL})
        result.amortized_setup_us.emplace_back(reuse, setup_ns / reuse / 1000.0);
    const auto drift = [](const std::vector<LifecycleSample>& values,
                          const PhaseTiming LifecycleSample::* phase) {
        if (values.size() < 2) return 0.0;
        const auto& first = values.front().*phase;
        const auto& last = values.back().*phase;
        if (!first.raw_ns || !last.raw_ns || *first.raw_ns == 0) return 0.0;
        return 100.0 * (static_cast<double>(*last.raw_ns) - *first.raw_ns) /
               static_cast<double>(*first.raw_ns);
    };
    result.allocation_latency_drift_percent =
        drift(result.samples, &LifecycleSample::backing_allocation);
    result.cleanup_latency_drift_percent = drift(result.samples, &LifecycleSample::cleanup);
    return result;
}

std::string Escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else if (c == '\n') out << "\\n";
        else if (c < 0x20) out << '?';
        else out << static_cast<char>(c);
    }
    return out.str();
}

void OptionalJson(std::ostringstream& out, const std::optional<std::uint64_t>& value) {
    if (value) out << *value; else out << "null";
}

std::string SnapshotJson(const MemorySnapshot& snapshot) {
    std::ostringstream out;
    out << "{\"installed_physical_bytes\":"; OptionalJson(out, snapshot.installed_physical_bytes);
    out << ",\"visible_physical_bytes\":"; OptionalJson(out, snapshot.visible_physical_bytes);
    out << ",\"available_physical_bytes\":"; OptionalJson(out, snapshot.available_physical_bytes);
    out << ",\"system_commit_bytes\":"; OptionalJson(out, snapshot.system_commit_bytes);
    out << ",\"system_commit_limit_bytes\":"; OptionalJson(out, snapshot.system_commit_limit_bytes);
    out << ",\"process_working_set_bytes\":"; OptionalJson(out, snapshot.process_working_set_bytes);
    out << ",\"process_private_bytes\":"; OptionalJson(out, snapshot.process_private_bytes);
    out << ",\"page_fault_count\":"; OptionalJson(out, snapshot.process_page_fault_count);
    out << ",\"memory_load_percent\":";
    if (snapshot.memory_load_percent) out << *snapshot.memory_load_percent; else out << "null";
    out << '}';
    return out.str();
}

std::string FormatBytes(std::uint64_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes < 1024ULL * 1024 * 1024)
        out << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    else
        out << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return out.str();
}

}  // namespace

TimerCalibration CalibrateTimer(std::uint32_t samples) {
    TimerCalibration calibration;
    calibration.raw_bracket_samples_ns.reserve(samples);
    std::uint64_t resolution = UINT64_MAX;
    for (std::uint32_t index = 0; index < samples; ++index) {
        const auto before = NowNs();
        const auto after = NowNs();
        const auto elapsed = after - before;
        calibration.raw_bracket_samples_ns.push_back(elapsed);
        if (elapsed > 0) resolution = std::min(resolution, elapsed);
    }
    std::vector<double> values;
    values.reserve(samples);
    for (const auto sample : calibration.raw_bracket_samples_ns)
        values.push_back(static_cast<double>(sample));
    calibration.bracket_overhead_ns = static_cast<std::uint64_t>(
        trace::CalculateDistribution(std::move(values)).median);
    calibration.observed_resolution_ns = resolution == UINT64_MAX ? 0 : resolution;
    return calibration;
}

std::uint32_t RepetitionsForSize(std::uint64_t bytes) noexcept {
    constexpr std::uint64_t mib = 1024ULL * 1024;
    if (bytes <= 64 * mib) return 9;
    if (bytes <= 256 * mib) return 7;
    if (bytes <= 1024 * mib) return 5;
    if (bytes <= 4096 * mib) return 3;
    return 2;
}

PhaseStatistics CalculatePhaseStatistics(
    const std::vector<LifecycleSample>& samples,
    const PhaseTiming LifecycleSample::* phase) {
    PhaseStatistics stats;
    std::vector<double> raw, corrected;
    for (const auto& sample : samples) {
        const auto& timing = sample.*phase;
        if (timing.raw_ns) raw.push_back(static_cast<double>(*timing.raw_ns));
        if (timing.corrected_ns) corrected.push_back(static_cast<double>(*timing.corrected_ns));
    }
    stats.count = raw.size();
    stats.raw_ns = trace::CalculateDistribution(std::move(raw));
    stats.corrected_ns = trace::CalculateDistribution(std::move(corrected));
    stats.p99_meaningful = stats.count >= 100;
    return stats;
}

LifecycleReport RunLifecycleLaboratory(IHostMemoryProvider& provider,
                                       const LifecycleOptions& options) {
    LifecycleReport report;
    report.timer = CalibrateTimer();
    report.initial_snapshot = provider.Snapshot();
    report.cuda_supported = provider.SupportsCudaHostMemory();
    bool needs_cuda = std::any_of(options.methods.begin(), options.methods.end(),
                                  [](auto method) { return method != MemoryMethod::Pageable; });
    if (needs_cuda && !options.dry_run) {
        report.cuda_warmup_performed = true;
        const auto start = NowNs();
        report.cuda_warmup_result = provider.WarmupCuda();
        report.cuda_warmup_ns = NowNs() - start;
    }

    for (const auto method : options.methods) {
        bool method_pressure = false;
        for (const auto bytes : options.sizes) {
            std::vector<RegistrationMode> modes{RegistrationMode::NotApplicable};
            if (method == MemoryMethod::CudaHostRegister) {
                modes.clear();
                if (options.include_cold_registration) modes.push_back(RegistrationMode::Cold);
                if (options.include_pretouched_registration)
                    modes.push_back(RegistrationMode::Pretouched);
            }
            for (const auto mode : modes) {
                const auto current = provider.Snapshot();
                const auto decision = PlanAllocation(current, options.safety, method, bytes,
                                                     method_pressure);
                if (!decision.safe()) {
                    LifecycleSample skipped;
                    skipped.method = method; skipped.mode = mode; skipped.requested_bytes = bytes;
                    skipped.status = ResultStatus::SkippedSafetyLimit;
                    skipped.safety = decision; skipped.before = current;
                    skipped.after_cleanup = current; skipped.message = decision.reason;
                    report.results.push_back(Aggregate(method, mode, bytes, {skipped}));
                    continue;
                }
                if (method != MemoryMethod::Pageable &&
                    (!provider.SupportsCudaHostMemory() || !report.cuda_warmup_result.ok())) {
                    LifecycleSample skipped;
                    skipped.method = method; skipped.mode = mode; skipped.requested_bytes = bytes;
                    skipped.status = provider.SupportsCudaHostMemory()
                                         ? report.cuda_warmup_result.status
                                         : ResultStatus::SkippedUnsupported;
                    skipped.safety = decision;
                    skipped.before = current; skipped.after_cleanup = current;
                    skipped.message = report.cuda_warmup_result.message;
                    report.results.push_back(Aggregate(method, mode, bytes, {skipped}));
                    continue;
                }
                if (options.dry_run) {
                    LifecycleSample planned;
                    planned.method = method; planned.mode = mode; planned.requested_bytes = bytes;
                    planned.status = ResultStatus::Success; planned.safety = decision;
                    planned.before = current; planned.after_cleanup = current;
                    planned.message = "dry run: no allocation attempted";
                    report.results.push_back(Aggregate(method, mode, bytes, {planned}));
                    continue;
                }
                const auto repetitions = options.repetitions.value_or(RepetitionsForSize(bytes));
                std::vector<LifecycleSample> samples;
                for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition) {
                    const auto repeated_decision = PlanAllocation(provider.Snapshot(), options.safety,
                                                                  method, bytes, method_pressure);
                    if (!repeated_decision.safe()) {
                        method_pressure = true;
                        LifecycleSample skipped;
                        skipped.method = method; skipped.mode = mode; skipped.requested_bytes = bytes;
                        skipped.repetition = repetition;
                        skipped.status = ResultStatus::AbortedPressureSignal;
                        skipped.safety = repeated_decision; skipped.message = repeated_decision.reason;
                        samples.push_back(std::move(skipped));
                        break;
                    }
                    auto sample = RunOne(provider, method, mode, bytes, repetition,
                                         repetition == 0, report.timer, repeated_decision, options);
                    const auto status = sample.status;
                    samples.push_back(std::move(sample));
                    if (ShouldStopAfter(status)) {
                        method_pressure = true;
                        report.pressure_stop = true;
                        report.pressure_stop_reason = samples.back().message;
                        break;
                    }
                }
                report.results.push_back(Aggregate(method, mode, bytes, std::move(samples)));
            }
        }
    }
    return report;
}

ArenaTestResult RunArenaTest(IHostMemoryProvider& provider,
                             ArenaBackend backend,
                             std::uint64_t bytes,
                             std::uint32_t reuse_count,
                             const SafetyPolicy& safety) {
    ArenaTestResult result;
    result.backend = backend; result.capacity_bytes = bytes; result.reuse_count = reuse_count;
    result.before = provider.Snapshot();
    const auto method = backend == ArenaBackend::CudaHostAlloc
                            ? MemoryMethod::CudaHostAlloc : MemoryMethod::CudaHostRegister;
    const auto decision = PlanAllocation(result.before, safety, method, bytes);
    if (!decision.safe()) {
        result.status = ResultStatus::SkippedSafetyLimit;
        result.message = decision.reason;
        return result;
    }
    if (!provider.SupportsCudaHostMemory()) {
        result.status = ResultStatus::SkippedUnsupported;
        result.message = "CUDA host memory is unavailable";
        return result;
    }
    const auto timer = CalibrateTimer();
    try {
        const auto start = NowNs();
        PersistentPinnedArena arena(provider, backend, bytes);
        const auto setup_raw = NowNs() - start;
        result.setup = {setup_raw, setup_raw > timer.bracket_overhead_ns
                                       ? setup_raw - timer.bracket_overhead_ns : 0};
        result.raw_reuse_ns.reserve(reuse_count);
        std::uint64_t checksum = 0;
        result.contents_verified = true;
        for (std::uint32_t reuse = 0; reuse < reuse_count; ++reuse) {
            const auto reuse_start = NowNs();
            auto view = arena.View(0, arena.capacity());
            const auto page = provider.PageSize();
            for (std::uint64_t offset = 0; offset < view.size(); offset += page) {
                view[static_cast<std::size_t>(offset)] =
                    static_cast<std::byte>((reuse + offset / page) & 0xFFU);
                checksum ^= static_cast<std::uint8_t>(view[static_cast<std::size_t>(offset)]);
            }
            for (std::uint64_t offset = 0; offset < view.size(); offset += page) {
                const auto expected = static_cast<std::byte>((reuse + offset / page) & 0xFFU);
                if (view[static_cast<std::size_t>(offset)] != expected)
                    result.contents_verified = false;
                checksum += static_cast<std::uint8_t>(view[static_cast<std::size_t>(offset)]);
            }
            result.raw_reuse_ns.push_back(NowNs() - reuse_start);
        }
        result.contents_verified = result.contents_verified && !result.raw_reuse_ns.empty();
        if (checksum == UINT64_MAX) result.contents_verified = false;
        if (!result.contents_verified) {
            result.status = ResultStatus::VerificationFailure;
            result.message = "persistent arena contents did not verify";
        }
        const auto cleanup_start = NowNs();
        const auto cleanup = arena.Close();
        const auto cleanup_raw = NowNs() - cleanup_start;
        result.cleanup = {cleanup_raw, cleanup_raw > timer.bracket_overhead_ns
                                           ? cleanup_raw - timer.bracket_overhead_ns : 0};
        if (!cleanup.ok()) {
            result.status = ResultStatus::CleanupFailure;
            result.message = cleanup.message;
        }
    } catch (const HostMemoryError& error) {
        result.status = error.result().status;
        result.message = error.what();
    } catch (const std::exception& error) {
        result.status = ResultStatus::InternalError;
        result.message = error.what();
    }
    std::vector<LifecycleSample> samples;
    for (const auto raw : result.raw_reuse_ns) {
        LifecycleSample sample;
        sample.warm_touch = {raw, raw > timer.bracket_overhead_ns
                                      ? raw - timer.bracket_overhead_ns : 0};
        samples.push_back(sample);
    }
    result.reuse = CalculatePhaseStatistics(samples, &LifecycleSample::warm_touch);
    result.after_cleanup = provider.Snapshot();
    return result;
}

std::string FormatMemoryInfo(const MemorySnapshot& snapshot) {
    std::ostringstream out;
    out << "SIDECAR HOST MEMORY INFORMATION\n\n"
        << "Installed: " << (snapshot.installed_physical_bytes
                                  ? FormatBytes(*snapshot.installed_physical_bytes) : "unknown") << '\n'
        << "OS visible: " << (snapshot.visible_physical_bytes
                                  ? FormatBytes(*snapshot.visible_physical_bytes) : "unknown") << '\n'
        << "Available: " << (snapshot.available_physical_bytes
                                  ? FormatBytes(*snapshot.available_physical_bytes) : "unknown") << '\n'
        << "System commit: " << (snapshot.system_commit_bytes
                                     ? FormatBytes(*snapshot.system_commit_bytes) : "unknown") << '\n'
        << "Commit limit: " << (snapshot.system_commit_limit_bytes
                                    ? FormatBytes(*snapshot.system_commit_limit_bytes) : "unknown") << '\n'
        << "Process working set: " << (snapshot.process_working_set_bytes
                                           ? FormatBytes(*snapshot.process_working_set_bytes) : "unknown") << '\n'
        << "Process private commit: " << (snapshot.process_private_bytes
                                             ? FormatBytes(*snapshot.process_private_bytes) : "unknown") << '\n'
        << "Memory load: " << (snapshot.memory_load_percent
                                   ? std::to_string(*snapshot.memory_load_percent) + "%" : "unknown") << '\n';
    return out.str();
}

std::string MemoryInfoToJson(const MemorySnapshot& snapshot) { return SnapshotJson(snapshot); }

std::string FormatMemoryPlan(const MemorySnapshot& snapshot,
                             const SafetyPolicy& policy,
                             MemoryMethod method,
                             const std::vector<std::uint64_t>& sizes) {
    std::ostringstream out;
    out << "SIDECAR HOST MEMORY PLAN\n\nMethod: " << ToString(method) << '\n'
        << "Available RAM: " << (snapshot.available_physical_bytes
                                     ? FormatBytes(*snapshot.available_physical_bytes) : "unknown") << '\n'
        << "Required reserve: " << FormatBytes(RequiredReserve(snapshot, policy)) << "\n\n";
    for (const auto bytes : sizes) {
        const auto decision = PlanAllocation(snapshot, policy, method, bytes);
        out << "  " << std::setw(10) << FormatBytes(bytes) << "  " << ToString(decision.status)
            << "  " << decision.reason << '\n';
    }
    return out.str();
}

std::string MemoryPlanToJson(const MemorySnapshot& snapshot,
                             const SafetyPolicy& policy,
                             MemoryMethod method,
                             const std::vector<std::uint64_t>& sizes) {
    std::ostringstream out;
    out << "{\"method\":\"" << ToString(method) << "\",\"snapshot\":"
        << SnapshotJson(snapshot) << ",\"required_reserve_bytes\":"
        << RequiredReserve(snapshot, policy) << ",\"tests\":[";
    for (std::size_t index = 0; index < sizes.size(); ++index) {
        if (index) out << ',';
        const auto decision = PlanAllocation(snapshot, policy, method, sizes[index]);
        out << "{\"requested_bytes\":" << sizes[index] << ",\"status\":\""
            << ToString(decision.status) << "\",\"projected_available_bytes\":"
            << decision.projected_available_bytes << ",\"policy_limit_bytes\":"
            << decision.method_policy_limit_bytes << ",\"reason\":\""
            << Escape(decision.reason) << "\"}";
    }
    out << "]}";
    return out.str();
}

std::string LifecycleReportToJson(const LifecycleReport& report) {
    std::ostringstream out;
    out << "{\"timer\":{\"method\":\"" << Escape(report.timer.method)
        << "\",\"bracket_overhead_ns\":" << report.timer.bracket_overhead_ns
        << ",\"resolution_ns\":" << report.timer.observed_resolution_ns
        << "},\"initial_snapshot\":" << SnapshotJson(report.initial_snapshot)
        << ",\"cuda_supported\":" << (report.cuda_supported ? "true" : "false")
        << ",\"cuda_warmup_performed\":"
        << (report.cuda_warmup_performed ? "true" : "false")
        << ",\"cuda_warmup_ns\":" << report.cuda_warmup_ns << ",\"results\":[";
    for (std::size_t index = 0; index < report.results.size(); ++index) {
        if (index) out << ',';
        const auto& result = report.results[index];
        out << "{\"method\":\"" << ToString(result.method) << "\",\"mode\":\""
            << ToString(result.mode) << "\",\"requested_bytes\":" << result.requested_bytes
            << ",\"status\":\"" << ToString(result.status) << "\",\"message\":\""
            << Escape(result.message) << "\",\"samples\":[";
        for (std::size_t sample_index = 0; sample_index < result.samples.size(); ++sample_index) {
            if (sample_index) out << ',';
            const auto& sample = result.samples[sample_index];
            const auto timing = [&](const char* name, const PhaseTiming& phase) {
                out << "\"" << name << "_raw_ns\":";
                if (phase.raw_ns) out << *phase.raw_ns; else out << "null";
                out << ",\"" << name << "_corrected_ns\":";
                if (phase.corrected_ns) out << *phase.corrected_ns; else out << "null";
            };
            out << "{\"repetition\":" << sample.repetition << ",\"cold_setup\":"
                << (sample.cold_setup ? "true" : "false") << ",\"status\":\""
                << ToString(sample.status) << "\",\"safety\":\""
                << ToString(sample.safety.status) << "\",";
            timing("allocation", sample.backing_allocation); out << ',';
            timing("first_touch", sample.first_touch); out << ',';
            timing("warm_touch", sample.warm_touch); out << ',';
            timing("register", sample.registration); out << ',';
            timing("unregister", sample.unregistration); out << ',';
            timing("cleanup", sample.cleanup);
            out << ",\"available_recovery_delta_bytes\":"
                << sample.available_recovery_delta_bytes
                << ",\"available_recovery_percent\":" << sample.available_recovery_percent
                << ",\"noisy\":"
                << (sample.noisy ? "true" : "false") << ",\"before\":"
                << SnapshotJson(sample.before) << ",\"after_setup\":"
                << SnapshotJson(sample.after_setup) << ",\"after_cleanup\":"
                << SnapshotJson(sample.after_cleanup) << ",\"cuda_error\":"
                << sample.cuda_error << ",\"windows_error\":" << sample.windows_error
                << ",\"message\":\"" << Escape(sample.message) << "\"}";
        }
        const auto statistics = [&](const char* name, const PhaseStatistics& phase) {
            out << "\"" << name << "\":{\"count\":" << phase.count
                << ",\"raw\":{\"mean_ns\":" << phase.raw_ns.mean
                << ",\"median_ns\":" << phase.raw_ns.median
                << ",\"p90_ns\":" << phase.raw_ns.p90
                << ",\"p95_ns\":" << phase.raw_ns.p95
                << ",\"p99_ns\":";
            if (phase.p99_meaningful) out << phase.raw_ns.p99; else out << "null";
            out << ",\"min_ns\":" << phase.raw_ns.minimum
                << ",\"max_ns\":" << phase.raw_ns.maximum
                << ",\"stddev_ns\":" << phase.raw_ns.stddev
                << "},\"corrected\":{\"mean_ns\":" << phase.corrected_ns.mean
                << ",\"median_ns\":" << phase.corrected_ns.median
                << ",\"p90_ns\":" << phase.corrected_ns.p90
                << ",\"p95_ns\":" << phase.corrected_ns.p95
                << ",\"p99_ns\":";
            if (phase.p99_meaningful) out << phase.corrected_ns.p99; else out << "null";
            out << ",\"min_ns\":" << phase.corrected_ns.minimum
                << ",\"max_ns\":" << phase.corrected_ns.maximum
                << ",\"stddev_ns\":" << phase.corrected_ns.stddev << "}}";
        };
        out << "],\"statistics\":{";
        statistics("allocation", result.allocation); out << ',';
        statistics("first_touch", result.first_touch); out << ',';
        statistics("warm_touch", result.warm_touch); out << ',';
        statistics("registration", result.registration); out << ',';
        statistics("unregistration", result.unregistration); out << ',';
        statistics("cleanup", result.cleanup);
        out << "},\"amortization\":[";
        for (std::size_t a = 0; a < result.amortized_setup_us.size(); ++a) {
            if (a) out << ',';
            out << "{\"conceptual_reuses\":" << result.amortized_setup_us[a].first
                << ",\"calculated_setup_us_per_reuse\":"
                << result.amortized_setup_us[a].second << '}';
        }
        out << "],\"allocation_drift_percent\":"
            << result.allocation_latency_drift_percent
            << ",\"cleanup_drift_percent\":"
            << result.cleanup_latency_drift_percent << '}';
    }
    out << "]}";
    return out.str();
}

std::string FormatLifecycleReport(const LifecycleReport& report) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "SIDECAR HOST MEMORY LIFECYCLE\n\nTimer\n"
        << "  Method: " << report.timer.method << '\n'
        << "  Empty bracket median: " << report.timer.bracket_overhead_ns << " ns\n"
        << "  Observed resolution: " << report.timer.observed_resolution_ns << " ns\n";
    if (report.cuda_warmup_performed)
        out << "  CUDA warmup: " << report.cuda_warmup_ns / 1000.0 << " us ("
            << ToString(report.cuda_warmup_result.status) << ")\n\n";
    else
        out << "  CUDA warmup: not run\n\n";
    for (const auto& result : report.results) {
        const auto stat = [&](const char* label, const PhaseStatistics& value) {
            if (!value.count) return;
            out << "    " << label << " median/p95 us: "
                << value.corrected_ns.median / 1000.0 << '/'
                << value.corrected_ns.p95 / 1000.0 << " count=" << value.count;
            if (!value.p99_meaningful) out << " P99=insufficient-samples";
            out << '\n';
        };
        out << "  " << ToString(result.method) << ' ' << ToString(result.mode)
            << " size=" << FormatBytes(result.requested_bytes)
            << " status=" << ToString(result.status) << '\n';
        stat("allocation", result.allocation);
        stat("first touch", result.first_touch);
        stat("warm touch", result.warm_touch);
        stat("register", result.registration);
        stat("unregister", result.unregistration);
        stat("cleanup", result.cleanup);
        if (!result.amortized_setup_us.empty())
            out << "    calculated setup/1000 reuses: "
                << result.amortized_setup_us[3].second << " us/reuse\n";
        out << "    first-to-last allocation/cleanup drift: "
            << result.allocation_latency_drift_percent << "% / "
            << result.cleanup_latency_drift_percent << "%\n";
        if (!result.message.empty()) out << "    message: " << result.message << '\n';
    }
    if (report.pressure_stop)
        out << "\nPressure stop: " << report.pressure_stop_reason << '\n';
    return out.str();
}

std::string ArenaResultToJson(const ArenaTestResult& result) {
    std::ostringstream out;
    out << "{\"backend\":\""
        << (result.backend == ArenaBackend::CudaHostAlloc ? "CUDA_HOST_ALLOC"
                                                         : "REGISTERED_PAGEABLE")
        << "\",\"capacity_bytes\":" << result.capacity_bytes
        << ",\"reuse_count\":" << result.reuse_count << ",\"status\":\""
        << ToString(result.status) << "\",\"verified\":"
        << (result.contents_verified ? "true" : "false")
        << ",\"setup_raw_ns\":" << result.setup.raw_ns.value_or(0)
        << ",\"setup_corrected_ns\":" << result.setup.corrected_ns.value_or(0)
        << ",\"cleanup_raw_ns\":" << result.cleanup.raw_ns.value_or(0)
        << ",\"cleanup_corrected_ns\":" << result.cleanup.corrected_ns.value_or(0)
        << ",\"reuse_median_ns\":" << result.reuse.raw_ns.median
        << ",\"reuse_p95_ns\":" << result.reuse.raw_ns.p95
        << ",\"before\":" << SnapshotJson(result.before)
        << ",\"after_cleanup\":" << SnapshotJson(result.after_cleanup)
        << ",\"raw_reuse_ns\":[";
    for (std::size_t index = 0; index < result.raw_reuse_ns.size(); ++index) {
        if (index) out << ',';
        out << result.raw_reuse_ns[index];
    }
    out << "],\"message\":\"" << Escape(result.message) << "\"}";
    return out.str();
}

std::string FormatArenaResult(const ArenaTestResult& result) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << "SIDECAR PERSISTENT PINNED ARENA\n\n"
        << "Backend: " << (result.backend == ArenaBackend::CudaHostAlloc
                                ? "cudaHostAlloc" : "registered pageable") << '\n'
        << "Capacity: " << FormatBytes(result.capacity_bytes) << '\n'
        << "Reuse count: " << result.reuse_count << '\n'
        << "Setup: " << result.setup.raw_ns.value_or(0) / 1000.0 << " us\n"
        << "Warm reuse median/P95: " << result.reuse.raw_ns.median / 1000.0 << '/'
        << result.reuse.raw_ns.p95 / 1000.0 << " us\n"
        << "Cleanup: " << result.cleanup.raw_ns.value_or(0) / 1000.0 << " us\n"
        << "Contents verified: " << (result.contents_verified ? "yes" : "no") << '\n'
        << "Status: " << ToString(result.status) << '\n';
    if (!result.message.empty()) out << "Message: " << result.message << '\n';
    return out.str();
}

}  // namespace sidecar::memory
