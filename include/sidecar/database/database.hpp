#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct sqlite3;

namespace sidecar::database {

inline constexpr std::int64_t kApplicationId = 0x53444352;
inline constexpr std::int64_t kCurrentSchemaVersion = 9;

class DatabaseError final : public std::runtime_error {
public:
    DatabaseError(int sqlite_code, const std::string& message);

    [[nodiscard]] int sqliteCode() const noexcept;

private:
    int sqlite_code_;
};

enum class OpenMode {
    CreateOrOpen,
    ExistingReadWrite,
};

struct DatabaseStatus {
    std::filesystem::path path;
    bool initialized{false};
    bool foreign_keys_enabled{false};
    std::int64_t application_id{0};
    std::int64_t schema_version{0};
    std::int64_t latest_migration{0};
    std::int64_t user_table_count{0};
    std::int64_t foreign_key_violations{0};
};

struct HardwareProfileInput {
    std::string machine_hash;
    std::string host_name;
    std::string os_name;
    std::string os_version;
    std::string os_build;
    std::string cpu_model;
    std::optional<std::int64_t> physical_core_count;
    std::optional<std::int64_t> logical_core_count;
    std::optional<std::int64_t> numa_node_count;
    std::optional<std::int64_t> installed_ram_bytes;
    std::optional<std::int64_t> available_ram_bytes;
    std::string discovery_json;
    std::int64_t identity_version{1};
    std::string identity_quality{"FALLBACK"};
    std::string identity_basis_json{"[]"};
    std::string system_manufacturer;
    std::string system_product;
    std::string smbios_version;
    std::string cpu_vendor;
    std::string cpu_architecture;
    std::optional<std::int64_t> processor_group_count;
    std::optional<std::int64_t> cpu_package_count;
    std::optional<std::int64_t> ram_speed_mt_s;
    std::string motherboard_manufacturer;
    std::string motherboard_model;
    std::string firmware_revision;
    std::string topology_json;
    std::string snapshot_json;
};

struct GpuDeviceInput {
    std::string persistent_id;
    std::string model;
    std::optional<std::int64_t> cuda_device_index;
    std::optional<std::int64_t> vram_bytes;
    std::optional<std::int64_t> pci_domain;
    std::optional<std::int64_t> pci_bus;
    std::optional<std::int64_t> pci_device;
    std::optional<std::int64_t> negotiated_pcie_generation;
    std::optional<std::int64_t> maximum_pcie_generation;
    std::optional<std::int64_t> negotiated_lane_width;
    std::optional<std::int64_t> maximum_lane_width;
    std::optional<std::int64_t> compute_capability_major;
    std::optional<std::int64_t> compute_capability_minor;
    std::optional<std::int64_t> async_engine_count;
    std::optional<std::int64_t> concurrent_kernels;
    std::optional<std::int64_t> unified_addressing;
    std::optional<std::int64_t> can_map_host_memory;
    std::optional<std::int64_t> managed_memory;
    std::optional<std::int64_t> pageable_memory_access;
    std::string attributes_json;
    std::string location_paths_json;
};

struct StorageDeviceInput {
    std::string persistent_id;
    std::string model;
    std::string firmware;
    std::optional<std::int64_t> capacity_bytes;
    std::string filesystem;
    std::string bus_type;
    std::string pci_path;
    std::optional<std::int64_t> numa_node;
    std::string topology_json;
    std::string volumes_json;
    std::string topology_confidence;
    std::string topology_source;
};

struct HardwareInventoryCounts {
    std::int64_t active_gpus{0};
    std::int64_t total_gpu_rows{0};
    std::int64_t active_storage_devices{0};
    std::int64_t total_storage_rows{0};
};

struct BenchmarkSessionInput {
    std::string machine_hash;
    std::string sidecar_spec_version;
    std::string sidecar_git_commit;
    std::optional<std::string> llama_cpp_git_commit;
    std::string trace_mode;
    std::optional<std::string> notes;
    std::optional<std::int64_t> cuda_runtime_version;
    std::optional<std::int64_t> cuda_driver_version;
    std::optional<std::string> cuda_toolkit_version;
    std::optional<std::string> nvidia_driver_version;
    std::optional<std::string> os_version;
};

struct TraceConfigurationInput {
    std::int64_t session_id{0};
    std::int64_t record_size_bytes{32};
    std::int64_t ring_capacity{65536};
    std::int64_t producer_count{1};
    std::int64_t collector_batch_size{4096};
    std::string collector_strategy{"spin_yield_sleep"};
    std::string timestamp_method{"steady_clock_ns"};
};

struct TraceBenchmarkInput {
    std::int64_t session_id{0};
    std::optional<std::int64_t> configuration_id;
    std::string workload;
    std::int64_t repetition{0};
    std::int64_t baseline_duration_ns{0};
    std::int64_t traced_duration_ns{0};
    std::int64_t events_generated{0};
    std::int64_t events_written{0};
    std::int64_t events_dropped{0};
    double events_per_second{0.0};
    std::optional<std::int64_t> producer_cpu_ns;
    std::optional<std::int64_t> collector_cpu_ns;
    double observer_overhead_raw{0.0};
    std::optional<std::int64_t> ring_high_water_mark;
    std::optional<std::int64_t> trace_bytes;
    std::optional<double> disk_write_bytes_per_second;
    std::optional<double> try_push_p50_ns;
    std::optional<double> try_push_p95_ns;
    std::optional<double> try_push_p99_ns;
    std::string raw_samples_json{"[]"};
};

struct TraceFileInput {
    std::int64_t session_id{0};
    std::string path;
    std::int64_t format_version{1};
    std::int64_t record_size_bytes{32};
    std::string trace_mode;
    std::int64_t trace_bytes{0};
    std::int64_t event_count{0};
    std::int64_t dropped_events{0};
    std::optional<std::int64_t> ring_high_water_mark;
    bool is_complete{false};
    std::optional<std::string> sha256;
};

struct HostMemoryBenchmarkInput {
    std::int64_t session_id{0};
    std::string method;
    std::string mode;
    std::int64_t requested_bytes{0};
    std::int64_t measured_repetitions{0};
    std::string status;
    std::string safety_status;
    std::string timer_method;
    std::int64_t timer_bracket_overhead_ns{0};
    std::int64_t timer_resolution_ns{0};
    std::optional<std::string> cuda_allocation_flags;
    std::optional<std::string> cuda_registration_flags;
    std::string statistics_json;
    std::string amortization_json;
    std::optional<std::string> message;
};

struct HostMemorySampleInput {
    std::int64_t benchmark_id{0};
    std::int64_t repetition{0};
    bool cold_setup{false};
    std::string status;
    std::optional<std::int64_t> allocation_raw_ns;
    std::optional<std::int64_t> allocation_corrected_ns;
    std::optional<std::int64_t> first_touch_raw_ns;
    std::optional<std::int64_t> first_touch_corrected_ns;
    std::optional<std::int64_t> warm_touch_raw_ns;
    std::optional<std::int64_t> warm_touch_corrected_ns;
    std::optional<std::int64_t> registration_raw_ns;
    std::optional<std::int64_t> registration_corrected_ns;
    std::optional<std::int64_t> unregistration_raw_ns;
    std::optional<std::int64_t> unregistration_corrected_ns;
    std::optional<std::int64_t> cleanup_raw_ns;
    std::optional<std::int64_t> cleanup_corrected_ns;
    std::string before_snapshot_json;
    std::string after_setup_snapshot_json;
    std::string after_cleanup_snapshot_json;
    std::optional<std::int64_t> available_recovery_delta_bytes;
    std::optional<double> available_recovery_percent;
    bool noisy{false};
    std::optional<std::int64_t> cuda_error;
    std::optional<std::int64_t> windows_error;
    std::optional<std::string> message;
};

struct PersistentArenaTestInput {
    std::int64_t session_id{0};
    std::string backend;
    std::int64_t capacity_bytes{0};
    std::int64_t reuse_count{0};
    std::optional<std::int64_t> setup_raw_ns;
    std::optional<std::int64_t> cleanup_raw_ns;
    std::string reuse_statistics_json;
    std::string raw_reuse_samples_json;
    bool contents_verified{false};
    std::string status;
    std::optional<std::string> message;
};

struct MemoryLaboratoryCounts {
    std::int64_t benchmarks{0};
    std::int64_t samples{0};
    std::int64_t pressure_events{0};
    std::int64_t arena_tests{0};
};

struct CudaTransferConfigurationInput {
    std::int64_t session_id{0};
    std::string configuration_hash, experiment, direction, host_memory_class, api_mode;
    std::int64_t transfer_bytes{0};
    std::string stream_mode;
    std::int64_t batch_count{1}, chunk_count{1}, warmup_count{0};
    std::int64_t measured_repetitions{0}, device_index{0};
    std::int64_t host_buffer_bytes{0}, device_buffer_bytes{0};
    std::string validation_mode, timer_mode;
    std::int64_t host_timer_bracket_overhead_ns{0}, host_timer_resolution_ns{0};
    std::string cuda_event_mode;
    std::int64_t ordering_seed{0};
    std::string status;
};

struct CudaTransferBenchmarkInput {
    std::int64_t configuration_id{0};
    std::string status;
    bool verified{false}, noisy{false};
    std::string async_behavior, statistics_json;
    std::optional<std::string> message;
};

struct CudaTransferSampleInput {
    std::int64_t benchmark_id{0}, repetition{0}, payload_bytes{0};
    std::int64_t host_api_raw_ns{0}, device_duration_ns{0}, end_to_end_raw_ns{0};
    std::string status;
    std::optional<std::int64_t> cuda_error;
    std::optional<std::string> message;
};

struct CudaBidirectionalBenchmarkInput {
    std::int64_t session_id{0};
    std::string host_memory_class;
    std::int64_t transfer_bytes{0}, measured_repetitions{0};
    double isolated_h2d_median_ns{0}, isolated_d2h_median_ns{0};
    double concurrent_h2d_median_ns{0}, concurrent_d2h_median_ns{0};
    double makespan_median_ns{0}, aggregate_bytes_per_second{0}, concurrency_benefit{0};
    bool verified{false};
    std::string status, raw_samples_json;
    std::optional<std::string> message;
};

struct CudaLinkStateInput {
    std::int64_t session_id{0};
    std::string phase;
    std::optional<std::int64_t> temperature_c, graphics_clock_mhz, memory_clock_mhz;
    std::optional<double> power_watts, power_limit_watts;
    std::optional<std::int64_t> pcie_generation, pcie_width;
};

struct CudaTransferProfileInput {
    std::int64_t session_id{0};
    std::string direction, host_memory_class, api_mode;
    double peak_bytes_per_second{0};
    std::int64_t peak_size_bytes{0}, knee_80_bytes{0}, knee_90_bytes{0}, knee_95_bytes{0};
    std::string candidate_sizes_json;
};

struct CudaTransferCounts {
    std::int64_t configurations{0}, benchmarks{0}, samples{0};
    std::int64_t bidirectional{0}, link_observations{0}, profiles{0};
};

struct CudaTransferProfileSummary {
    std::int64_t session_id{0};
    std::string direction, host_memory_class, api_mode;
    double peak_bytes_per_second{0};
    std::int64_t peak_size_bytes{0}, knee_80_bytes{0}, knee_90_bytes{0}, knee_95_bytes{0};
    double host_api_median_ns{0}, device_median_ns{0}, end_to_end_median_ns{0};
    double device_p95_ns{0}, device_p99_ns{0};
    bool p99_meaningful{false};
};

struct CudaTransferExperimentSummary {
    std::int64_t session_id{0};
    std::string experiment, direction, host_memory_class;
    std::int64_t transfer_bytes{0}, batch_count{0}, chunk_count{0}, sample_count{0};
    double mean_host_api_ns{0}, mean_device_ns{0}, mean_end_to_end_ns{0};
    double mean_bytes_per_second{0};
};

struct CudaSustainedSummary {
    std::int64_t session_id{0};
    std::string direction, host_memory_class;
    std::int64_t transfer_bytes{0}, payload_bytes{0};
    double bytes_per_second{0};
};

struct CudaLinkStateSummary {
    std::int64_t session_id{0};
    std::string phase;
    std::optional<std::int64_t> temperature_c, graphics_clock_mhz, memory_clock_mhz;
    std::optional<double> power_watts, power_limit_watts;
    std::optional<std::int64_t> pcie_generation, pcie_width;
};

struct CudaTransferReportData {
    CudaTransferCounts counts;
    std::vector<CudaTransferProfileSummary> latest_profiles;
    std::vector<CudaTransferExperimentSummary> latest_supplemental;
    std::optional<CudaSustainedSummary> best_sustained;
    std::vector<CudaLinkStateSummary> latest_link_states;
    double best_bidirectional_bytes_per_second{0};
};

struct ComputeWorkloadProfileInput {
    std::int64_t session_id{0};
    std::string workload;
    double target_compute_us{0};
    std::string calibration_statistics_json;
    std::int64_t calibration_samples{0}, alu_iterations{0};
    std::int64_t memory_working_set_bytes{0}, memory_passes{0};
    std::int64_t memory_block_size{0}, memory_elements_per_thread{0};
    std::int64_t gemm_m{0}, gemm_n{0}, gemm_k{0}, gemm_repetitions{0};
    std::string gemm_a_type, gemm_b_type, gemm_c_type, gemm_compute_type;
    std::string gemm_math_mode, gemm_algorithm;
    std::optional<std::int64_t> cublas_version;
    bool validated{false};
    std::string status;
    std::optional<std::string> message;
};

struct CudaOverlapConfigurationInput {
    std::int64_t session_id{0}, compute_workload_profile_id{0};
    std::string configuration_hash, phase, direction, host_memory_class;
    std::int64_t transfer_bytes{0};
    double target_compute_us{0};
    std::int64_t measured_repetitions{0}, ordering_seed{0};
    std::int64_t gate_delay_ns{0}, gate_margin_ns{0}, device_index{0};
    std::string instrumentation_mode, refinement_reason, status;
};

struct CudaOverlapBenchmarkInput {
    std::int64_t configuration_id{0};
    std::string status, statistics_json, telemetry_before_json,
        telemetry_after_json;
    bool refined{false};
    std::optional<std::string> message;
};

struct CudaOverlapSampleInput {
    std::int64_t benchmark_id{0}, repetition{0}, baseline_block_id{0};
    std::int64_t c0_reference_ns{0}, t0_reference_ns{0}, cc_ns{0}, tc_ns{0};
    std::int64_t makespan_device_primary_ns{0};
    std::int64_t makespan_device_crosscheck_ns{0}, makespan_host_ns{0};
    std::int64_t host_submission_ns{0}, gate_delay_ns{0}, gate_actual_ns{0},
        gate_margin_ns{0};
    double critical_path_delta_ns{0}, compute_path_added_ns{0};
    double compute_path_added_percent{0}, compute_slowdown{0};
    double transfer_slowdown{0}, overlap_efficiency_raw{0};
    double overlap_efficiency_normalized{0}, hidden_fraction_raw{0};
    double hidden_fraction_normalized{0}, compute_retention{0};
    bool gate_valid{false}, fit_1_percent{false}, fit_2_percent{false},
        fit_5_percent{false};
    std::string instrumentation_status, status;
    std::optional<std::int64_t> native_error;
    std::optional<std::string> message;
};

struct CudaOverlapFitProfileInput {
    std::int64_t session_id{0};
    std::string workload, direction, host_memory_class;
    double target_compute_us{0}, tolerance_percent{0};
    std::int64_t largest_measured_bytes{0};
    double p99_added_percent{0}, fit_rate{0};
    std::string confidence;
};

struct InstrumentationControlInput {
    std::int64_t session_id{0};
    std::string workload, direction, instrumentation_mode;
    std::int64_t transfer_bytes{0};
    double target_compute_us{0};
    std::string compute_statistics_json, transfer_statistics_json,
        host_statistics_json;
    double compute_bias_percent{0}, transfer_bias_percent{0};
    bool rejected{false};
    std::string status;
};

struct CudaOverlapCounts {
    std::int64_t profiles{0}, configurations{0}, benchmarks{0}, samples{0};
    std::int64_t fit_profiles{0}, instrumentation_controls{0};
};

struct CudaOverlapEnvelopeSummary {
    std::int64_t session_id{0};
    std::string workload, direction, host_memory_class;
    double target_compute_us{0}, tolerance_percent{0};
    std::int64_t largest_measured_bytes{0};
    double p99_added_percent{0}, fit_rate{0};
    std::string confidence;
};

struct StorageDatasetInput {
    std::string machine_hash, persistent_id, generator, identity_hash, file_path, volume_name;
    std::int64_t format_version{1}, seed{0}, size_bytes{0}, physical_disk_number{0};
    bool verified{false}, full_verification{false};
    std::int64_t verified_bytes{0};
};

struct StorageHealthSnapshotInput {
    std::string machine_hash, persistent_id, phase, provider, status, raw_evidence_json;
    std::optional<std::int64_t> session_id;
    std::optional<std::int64_t> critical_warning, available_spare_percent;
    std::optional<std::int64_t> available_spare_threshold_percent, percentage_used;
    std::optional<std::int64_t> data_units_read_low64, data_units_written_low64;
    std::optional<std::int64_t> host_read_commands_low64, host_write_commands_low64;
    std::optional<std::int64_t> controller_busy_minutes_low64, power_cycles_low64;
    std::optional<std::int64_t> power_on_hours_low64, unsafe_shutdowns_low64;
    std::optional<std::int64_t> media_data_errors_low64, error_log_entries_low64;
    std::optional<double> temperature_c;
    std::optional<std::string> message;
};

struct StorageConfigurationInput {
    std::int64_t session_id{0}, storage_dataset_id{0};
    std::string configuration_hash, backend, access_pattern, destination_kind, phase;
    std::string cache_classification, status;
    std::int64_t block_bytes{0}, queue_depth{0}, request_count{0};
    std::int64_t outstanding_byte_cap{0}, timeout_ms{0}, ordering_seed{0}, window_bytes{0};
    std::optional<std::string> skip_reason;
};

struct StorageBenchmarkInput {
    std::int64_t configuration_id{0};
    std::optional<std::int64_t> health_before_id, health_after_id;
    std::string status, statistics_json, cpu_json;
    bool verified{false};
    std::int64_t completed_bytes{0}, wall_time_ns{0};
    double bytes_per_second{0}, iops{0};
    std::optional<std::string> message;
};

struct StorageSampleInput {
    std::int64_t benchmark_id{0}, request_index{0}, batch_id{0}, file_offset{0};
    std::int64_t requested_bytes{0}, completed_bytes{0}, submission_reference_ns{0};
    std::int64_t submission_cost_ns{0}, completion_latency_ns{0};
    std::int64_t completion_processing_ns{0};
    std::string status;
    std::optional<std::int64_t> native_error;
};

struct StorageDeadlineInput {
    std::int64_t benchmark_id{0}, deadline_ns{0}, hits{0}, misses{0};
    double success_rate{0}, compliant_bytes_per_second{0};
};

struct StorageBackendProfileInput {
    std::int64_t session_id{0};
    std::string backend, access_pattern, destination_kind;
    double peak_bytes_per_second{0};
    std::int64_t peak_block_bytes{0}, peak_queue_depth{0};
    std::optional<std::int64_t> knee_80_queue_depth, knee_90_queue_depth;
    std::optional<std::int64_t> knee_95_queue_depth;
    std::string safe_envelopes_json{"{}"}, supply_profile_json{"{}"};
};

struct StoragePhysicsCounts {
    std::int64_t datasets{0}, health_snapshots{0}, configurations{0};
    std::int64_t benchmarks{0}, samples{0}, deadline_profiles{0}, backend_profiles{0};
};

struct StorageBenchmarkSummary {
    std::int64_t session_id{0}, block_bytes{0}, queue_depth{0}, sample_count{0};
    std::string backend, access_pattern, destination_kind, status;
    double bytes_per_second{0}, iops{0}, p50_ns{0}, p95_ns{0}, p99_ns{0};
};

struct HostCopyConfigurationInput {
    std::int64_t session_id{0};
    std::string configuration_hash;
    std::int64_t bytes{0}, worker_count{0}, warmups{0}, repetitions{0};
    std::optional<std::string> concurrent_compute;
    double compute_window_us{0};
    std::string phase, status;
};

struct HostCopyBenchmarkInput {
    std::int64_t configuration_id{0};
    std::string status, statistics_json, affinity;
    std::optional<std::string> message;
};

struct HostCopySampleInput {
    std::int64_t benchmark_id{0}, sample_index{0}, wall_ns{0}, process_cpu_ns{0},
        compute_ns{0};
    double bytes_per_second{0};
    std::string status;
    std::optional<std::string> message;
};

struct PipelineConfigurationInput {
    std::int64_t session_id{0}, storage_dataset_id{0};
    std::string configuration_hash, pipeline_type;
    std::int64_t aggregate_bytes{0}, chunk_bytes{0}, chunk_count{0}, buffer_depth{0};
    std::string compute_type;
    double compute_window_us{0};
    std::int64_t deadline_ns{0}, measured_repetitions{0}, ordering_seed{0}, timeout_ms{0};
    std::string phase, contention_test;
    std::int64_t host_copy_workers{0}, device_index{0};
    std::string arena_allocation_id;
    std::optional<std::string> refinement_reason;
    std::string status;
};

struct PipelineBenchmarkInput {
    std::int64_t configuration_id{0};
    std::optional<std::int64_t> health_before_id, health_after_id;
    std::string status, statistics_json, gpu_before_json, gpu_after_json,
        host_before_json, host_after_json;
    std::optional<std::string> message;
};

struct PipelineSampleInput {
    std::int64_t benchmark_id{0}, sample_index{0};
    std::int64_t c0_reference_ns{0}, p0_reference_ns{0}, cc_ns{0}, pc_ns{0},
        makespan_ns{0};
    double compute_path_added_ns{0}, compute_path_added_percent{0};
    double pipeline_slowdown{0}, compute_slowdown{0};
    double pipeline_overlap_raw{0}, pipeline_overlap_normalized{0};
    std::int64_t deadline_ns{0}, ready_ahead_ns{0};
    bool deadline_hit{false};
    std::int64_t storage_ns{0}, host_copy_ns{0}, h2d_ns{0}, fill_latency_ns{0};
    std::int64_t steady_state_interval_ns{0}, drain_latency_ns{0};
    std::int64_t gpu_waiting_for_data_ns{0}, h2d_waiting_for_source_ns{0};
    std::int64_t pinned_slot_wait_ns{0}, pageable_slot_wait_ns{0},
        nvme_queue_starved_ns{0};
    std::string slot_ids_json;
    bool verified{false};
    std::string status;
    std::optional<std::int64_t> native_error;
    std::optional<std::string> message;
};

struct PipelineStageSampleInput {
    std::int64_t benchmark_id{0}, sample_index{0}, block_id{0}, slot_id{0};
    std::string stage_type;
    std::int64_t bytes{0}, file_offset{0}, host_submit_ns{0}, host_start_ns{0},
        host_finish_ns{0}, device_duration_ns{0}, wait_ns{0};
    std::string status;
    std::optional<std::int64_t> native_error;
};

struct PipelineDeadlineProfileInput {
    std::int64_t benchmark_id{0}, deadline_ns{0}, hits{0}, misses{0};
    double success_rate{0};
    std::string ready_ahead_statistics_json, lateness_statistics_json;
    bool p999_supported{false};
};

struct PipelineSlotProfileInput {
    std::int64_t benchmark_id{0}, session_id{0};
    std::string machine_hash, arena_allocation_id;
    std::int64_t slot_id{0}, arena_offset{0}, bytes{0}, transfer_count{0};
    std::int64_t verification_failures{0}, deadline_hits{0}, deadline_misses{0};
    std::string host_copy_statistics_json, h2d_statistics_json, observation;
};

struct PipelineHealthObservationInput {
    std::int64_t benchmark_id{0};
    std::string phase;
    std::optional<std::int64_t> nvme_health_snapshot_id;
    std::string gpu_telemetry_json, host_memory_json, process_cpu_json, status;
};

struct PipelinePhysicsCounts {
    std::int64_t host_copy_configurations{0}, host_copy_benchmarks{0},
        host_copy_samples{0}, pipeline_configurations{0}, pipeline_benchmarks{0},
        pipeline_samples{0}, pipeline_stage_samples{0}, deadline_profiles{0},
        slot_profiles{0}, health_observations{0};
};

struct PipelineBenchmarkSummary {
    std::int64_t session_id{0}, aggregate_bytes{0}, chunk_bytes{0}, buffer_depth{0},
        sample_count{0};
    std::string pipeline_type, compute_type, contention_test, phase, status;
    double compute_window_us{0}, p50_ns{0}, p95_ns{0}, p99_ns{0};
    double deadline_success_64ms{0}, p99_compute_added_percent{0};
};

struct LlamaDependencyInput {
    std::string dependency_name{"llama.cpp"};
    std::string upstream_url;
    std::string upstream_commit;
    bool dirty{false};
    std::string build_configuration;
    bool cuda_enabled{false};
    std::string compiler;
    std::string sidecar_git_commit;
    std::string binary_identity_json{"{}"};
};

struct GgufModelInput {
    std::string model_id;
    std::string source;
    std::optional<std::string> source_repository;
    std::optional<std::string> source_revision;
    std::optional<std::string> ollama_model_tag;
    std::optional<std::string> ollama_digest;
    std::optional<std::string> original_blob_path;
    std::optional<std::string> alias_path;
    std::string local_path;
    std::string sha256;
    std::int64_t file_size_bytes{0};
    std::int64_t gguf_version{0};
    std::optional<std::string> architecture;
    std::optional<std::string> quantization;
    std::optional<std::int64_t> parameter_count;
    std::int64_t tensor_count{0};
    std::optional<std::int64_t> layer_count;
    std::optional<std::int64_t> expert_count;
    std::string metadata_json{"{}"};
    std::string verification_status;
    std::optional<std::string> verification_detail;
    std::string llama_cpp_commit;
};

struct GgufTensorInput {
    std::string model_id;
    std::int64_t tensor_id{0};
    std::string tensor_name;
    std::string tensor_type;
    std::string dimensions_json;
    std::int64_t tensor_bytes{0};
    std::int64_t file_offset{0};
    std::int64_t file_span_bytes{0};
    std::int64_t alignment_bytes{0};
    std::optional<std::int64_t> layer_index;
    std::optional<std::int64_t> expert_index;
    std::string tensor_role;
    bool persistent_weight{true};
};

struct GgufModelStorageInput {
    std::string model_id, requested_path, resolved_path, volume_path;
    std::optional<std::string> volume_unique_id;
    std::string physical_device;
    std::optional<std::int64_t> physical_disk_number;
    std::string device_model, bus_type, storage_role;
    bool is_samsung_990_pro{false}, is_usb_external{false};
    std::optional<std::string> copy_relationship;
};

struct LlamaObservationCounts {
    std::int64_t dependencies{0};
    std::int64_t models{0};
    std::int64_t tensors{0};
    std::int64_t storage_provenance{0};
    std::int64_t configurations{0};
    std::int64_t benchmarks{0};
    std::int64_t token_samples{0};
    std::int64_t observer_events{0};
    std::int64_t tensor_demand_events{0};
    std::int64_t block_projections{0};
    std::int64_t shadow_results{0};
    std::int64_t overhead_benchmarks{0};
    std::int64_t moe_profiles{0};
};

struct InferenceFixtureInput {
    std::string fixture_id, fixture_name, text_sha256, description;
    std::int64_t fixture_version{1}, text_bytes{0};
    std::optional<std::int64_t> token_count;
    std::optional<std::string> token_ids_json;
};

struct InferenceConfigurationRecordInput {
    std::int64_t session_id{0};
    std::string model_id;
    std::optional<std::string> fixture_id;
    std::string configuration_hash, phase, observer_mode;
    std::int64_t gpu_layers{-1}, context_size{0}, prompt_tokens{0}, generated_tokens{0};
    std::int64_t warmups{0}, repetitions{0}, seed{0};
    std::string sampling_json{"{\"type\":\"greedy\"}"};
    std::string backend_mode{"CUDA"}, load_mode{"MMAP"}, status{"COMPLETE"};
};

struct InferenceBenchmarkRecordInput {
    std::int64_t configuration_id{0};
    std::string status;
    std::optional<std::int64_t> model_load_ns;
    std::string prompt_statistics_json{"{}"}, decode_statistics_json{"{}"};
    std::string perf_context_json{"{}"}, telemetry_json{"{}"};
    std::string graph_split_detail{"BACKEND_DETAIL_UNAVAILABLE_PUBLIC_API"};
    std::optional<std::string> message;
};

struct InferenceSampleRecordInput {
    std::int64_t benchmark_id{0}, sample_index{0};
    std::int64_t prompt_ns{0}, decode_total_ns{0};
    double prompt_tokens_per_second{0}, generation_tokens_per_second{0};
    std::int64_t graph_node_count{0};
    std::optional<std::int64_t> graph_split_count;
    std::int64_t observer_event_count{0}, dropped_event_count{0};
    std::string status{"PASS"};
};

struct InferenceTokenRecordInput {
    std::int64_t benchmark_id{0}, sample_index{0};
    std::string phase{"DECODE"};
    std::int64_t token_index{0}, input_token_id{0};
    std::optional<std::int64_t> output_token_id;
    std::int64_t context_depth{0}, decode_start_ns{0}, decode_end_ns{0}, decode_duration_ns{0};
    std::string observer_mode;
    std::int64_t graph_node_count{0};
    std::optional<std::string> layer_sequence_reference;
    std::int64_t unique_weight_tensor_count{0}, unique_weight_bytes{0};
    std::string projected_blocks_json{"{}"};
    std::optional<std::string> telemetry_reference;
    std::string status{"PASS"};
};

struct ObserverProfileRecordInput {
    std::int64_t benchmark_id{0};
    std::string observer_mode;
    std::int64_t ask_calls{0}, materialized_calls{0}, compact_events{0}, dropped_events{0};
    std::optional<std::int64_t> ring_high_water_mark;
    std::string callback_contract, dictionary_json{"{}"};
};

struct ObserverEventRecordInput {
    std::int64_t observer_profile_id{0}, event_index{0}, timestamp_ns{0};
    std::int64_t eval_index{0}, sequence_index{0};
    std::optional<std::int64_t> tensor_id_0, tensor_id_1;
    std::int64_t op_id{0};
    std::optional<std::int64_t> layer_index;
    std::int64_t event_class{0}, flags{0};
};

struct TensorDemandRecordInput {
    std::int64_t benchmark_id{0};
    std::int64_t token_index{0}, sequence_index{0}, tensor_id{0}, first_event_index{0};
    std::optional<std::int64_t> layer_index;
    std::int64_t tensor_bytes{0}, file_offset{0};
    std::optional<std::int64_t> causal_known_at_ns;
    std::int64_t demand_ns{0};
};

struct TensorBlockProjectionRecordInput {
    std::optional<std::int64_t> benchmark_id;
    std::string model_id;
    std::int64_t block_size_bytes{0};
    std::string scope;
    std::optional<std::int64_t> scope_index;
    std::int64_t useful_tensor_bytes{0}, projected_block_count{0}, projected_bytes{0}, overfetch_bytes{0};
    double overfetch_ratio{0};
    std::string block_ids_json{"[]"};
};

struct ObserverOverheadRecordInput {
    std::int64_t session_id{0};
    std::string model_id, phase, observer_mode;
    std::int64_t paired_repetitions{0};
    std::string baseline_samples_json, observed_samples_json, paired_delta_statistics_json;
    double overhead_percent{0};
    std::string confidence_interval_json;
    double production_gate_percent{2};
    bool gate_passed{false};
};

struct MoeDemandProfileRecordInput {
    std::optional<std::int64_t> inference_benchmark_id;
    std::string model_id, status;
    std::optional<std::int64_t> layer_index, expert_count, expert_working_set_bytes;
    bool selected_experts_observable{false};
    std::string reuse_json{"{}"}, causal_cue_json{"{}"};
};

struct ShadowPipelineRecordInput {
    std::int64_t inference_benchmark_id{0}, token_index{0}, tensor_or_block_id{0};
    std::int64_t block_size_bytes{0};
    std::string pipeline_type, lookahead_assumption, knowledge_mode;
    std::int64_t available_lead_time_ns{0};
    std::string pipeline_profile_reference;
    std::int64_t deadline_ns{0};
    bool predicted_hit{false};
    std::int64_t margin_ns{0};
    bool extrapolated{false};
};

class Database final {
public:
    static Database Open(const std::filesystem::path& path, OpenMode mode);

    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&& other) noexcept;
    Database& operator=(Database&& other) noexcept;

    void Initialize();
    [[nodiscard]] DatabaseStatus Status() const;

    void UpsertHardwareProfile(const HardwareProfileInput& profile);
    void RefreshHardwareInventory(const HardwareProfileInput& profile,
                                  const std::vector<GpuDeviceInput>& gpus,
                                  const std::vector<StorageDeviceInput>& storage_devices,
                                  bool refresh_gpu_presence = true,
                                  bool refresh_storage_presence = true);
    [[nodiscard]] HardwareInventoryCounts InventoryCounts(
        const std::string& machine_hash) const;
    [[nodiscard]] std::int64_t StartBenchmarkSession(const BenchmarkSessionInput& session);
    void CompleteBenchmarkSession(std::int64_t session_id, const std::string& status);
    [[nodiscard]] std::int64_t InsertTraceConfiguration(
        const TraceConfigurationInput& configuration);
    [[nodiscard]] std::int64_t InsertTraceBenchmark(const TraceBenchmarkInput& benchmark);
    [[nodiscard]] std::int64_t InsertTraceFile(const TraceFileInput& trace_file);
    [[nodiscard]] std::int64_t InsertHostMemoryBenchmark(
        const HostMemoryBenchmarkInput& benchmark);
    void InsertHostMemorySample(const HostMemorySampleInput& sample);
    [[nodiscard]] std::int64_t InsertPersistentArenaTest(
        const PersistentArenaTestInput& test);
    [[nodiscard]] MemoryLaboratoryCounts MemoryCounts() const;
    [[nodiscard]] std::int64_t InsertCudaTransferConfiguration(
        const CudaTransferConfigurationInput& configuration);
    [[nodiscard]] std::int64_t InsertCudaTransferBenchmark(
        const CudaTransferBenchmarkInput& benchmark);
    void InsertCudaTransferSample(const CudaTransferSampleInput& sample);
    [[nodiscard]] std::int64_t InsertCudaBidirectionalBenchmark(
        const CudaBidirectionalBenchmarkInput& benchmark);
    void InsertCudaLinkState(const CudaLinkStateInput& observation);
    void InsertCudaTransferProfile(const CudaTransferProfileInput& profile);
    [[nodiscard]] CudaTransferCounts TransferCounts() const;
    [[nodiscard]] CudaTransferReportData TransferReport() const;
    [[nodiscard]] std::int64_t InsertComputeWorkloadProfile(
        const ComputeWorkloadProfileInput& profile);
    [[nodiscard]] std::int64_t InsertCudaOverlapConfiguration(
        const CudaOverlapConfigurationInput& configuration);
    [[nodiscard]] std::int64_t InsertCudaOverlapBenchmark(
        const CudaOverlapBenchmarkInput& benchmark);
    void InsertCudaOverlapSample(const CudaOverlapSampleInput& sample);
    void InsertCudaOverlapFitProfile(const CudaOverlapFitProfileInput& profile);
    void InsertInstrumentationControl(const InstrumentationControlInput& control);
    [[nodiscard]] CudaOverlapCounts OverlapCounts() const;
    [[nodiscard]] std::vector<CudaOverlapEnvelopeSummary>
    LatestOverlapEnvelopes() const;
    [[nodiscard]] std::int64_t UpsertStorageDataset(const StorageDatasetInput& input);
    [[nodiscard]] std::int64_t InsertStorageHealthSnapshot(
        const StorageHealthSnapshotInput& input);
    [[nodiscard]] std::int64_t InsertStorageConfiguration(
        const StorageConfigurationInput& input);
    [[nodiscard]] std::int64_t InsertStorageBenchmark(const StorageBenchmarkInput& input);
    void InsertStorageSample(const StorageSampleInput& input);
    void InsertStorageDeadline(const StorageDeadlineInput& input);
    void InsertStorageBackendProfile(const StorageBackendProfileInput& input);
    [[nodiscard]] StoragePhysicsCounts StorageCounts() const;
    [[nodiscard]] std::vector<StorageBenchmarkSummary> LatestStorageBenchmarks() const;
    [[nodiscard]] std::int64_t InsertHostCopyConfiguration(
        const HostCopyConfigurationInput& input);
    [[nodiscard]] std::int64_t InsertHostCopyBenchmark(const HostCopyBenchmarkInput& input);
    void InsertHostCopySample(const HostCopySampleInput& input);
    [[nodiscard]] std::int64_t InsertPipelineConfiguration(
        const PipelineConfigurationInput& input);
    [[nodiscard]] std::int64_t InsertPipelineBenchmark(const PipelineBenchmarkInput& input);
    void InsertPipelineSample(const PipelineSampleInput& input);
    void InsertPipelineStageSample(const PipelineStageSampleInput& input);
    void InsertPipelineDeadlineProfile(const PipelineDeadlineProfileInput& input);
    void InsertPipelineSlotProfile(const PipelineSlotProfileInput& input);
    void InsertPipelineHealthObservation(const PipelineHealthObservationInput& input);
    [[nodiscard]] PipelinePhysicsCounts PipelineCounts() const;
    [[nodiscard]] std::vector<PipelineBenchmarkSummary> LatestPipelineBenchmarks() const;
    [[nodiscard]] std::int64_t UpsertLlamaDependency(const LlamaDependencyInput& input);
    void UpsertGgufModel(const GgufModelInput& input);
    void ReplaceGgufTensors(const std::string& model_id,
                            const std::vector<GgufTensorInput>& tensors);
    void UpsertGgufModelStorage(const GgufModelStorageInput& input);
    void UpsertInferenceFixture(const InferenceFixtureInput& input);
    [[nodiscard]] std::int64_t InsertInferenceConfiguration(
        const InferenceConfigurationRecordInput& input);
    [[nodiscard]] std::int64_t InsertInferenceBenchmark(
        const InferenceBenchmarkRecordInput& input);
    void InsertInferenceSample(const InferenceSampleRecordInput& input);
    void InsertInferenceToken(const InferenceTokenRecordInput& input);
    [[nodiscard]] std::int64_t InsertObserverProfile(const ObserverProfileRecordInput& input);
    void InsertObserverEvent(const ObserverEventRecordInput& input);
    void UpsertTensorDemandEvent(const TensorDemandRecordInput& input);
    void InsertTensorBlockProjection(const TensorBlockProjectionRecordInput& input);
    void InsertObserverOverhead(const ObserverOverheadRecordInput& input);
    void InsertMoeDemandProfile(const MoeDemandProfileRecordInput& input);
    [[nodiscard]] std::optional<std::int64_t> LatestInferenceBenchmarkId(
        const std::string& model_id) const;
    void InsertShadowPipelineResult(const ShadowPipelineRecordInput& input);
    [[nodiscard]] LlamaObservationCounts LlamaCounts() const;
    void BeginWriteTransaction();
    void CommitWriteTransaction();
    void RollbackWriteTransaction() noexcept;

private:
    Database(sqlite3* handle, std::filesystem::path path) noexcept;

    void Execute(const std::string& sql) const;
    [[nodiscard]] std::int64_t QueryInt64(const std::string& sql) const;
    [[nodiscard]] bool TableExists(const std::string& table_name) const;

    sqlite3* handle_{nullptr};
    std::filesystem::path path_;
};

}  // namespace sidecar::database
